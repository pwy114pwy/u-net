"""
=============================================================================
模块 5: 模型评估 (Evaluation)
论文: 基于深度卷积神经网络的水下偏色图像增强方法

评估指标：
    PSNR  (峰值信噪比)  —— 越高越好，衡量像素级别的误差
    SSIM  (结构相似度)  —— 越接近 1 越好，衡量结构/亮度/对比度
=============================================================================

使用方法：
    python evaluate.py \
        --val_input_dir  data/val/input \
        --val_target_dir data/val/target \
        --ckpt           checkpoints/best_model.pth

=============================================================================
"""

import argparse
import math
import os
from pathlib import Path

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from model import UNet


# =============================================================================
# PSNR 计算
# =============================================================================

def compute_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """
    计算单对图像的 PSNR（峰值信噪比）。

    Args:
        pred   : 模型预测图像张量，值域 [0, 1]，形状 [C, H, W]
        target : 参考标准图像张量，值域 [0, 1]，形状 [C, H, W]
        max_val: 像素值的最大值（归一化后为 1.0）

    Returns:
        PSNR 值（单位 dB），float。若两张图完全一致则返回 inf。
    """
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10 * math.log10((max_val ** 2) / mse)


# =============================================================================
# SSIM 计算（纯 PyTorch 实现，无需额外依赖）
# =============================================================================

def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """生成一维高斯核，用于 SSIM 的滑窗卷积。"""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> float:
    """
    计算单对图像的 SSIM（结构相似度索引）。

    基于论文：
        Z. Wang et al., "Image quality assessment: from error visibility to
        structural similarity," IEEE TIP, 2004.

    Args:
        pred       : 模型预测图像，值域 [0, 1]，形状 [C, H, W]
        target     : 参考图像，值域 [0, 1]，形状 [C, H, W]
        window_size: 高斯滑窗尺寸（论文默认 11）
        sigma      : 高斯分布标准差（论文默认 1.5）
        C1, C2     : 数值稳定性常数

    Returns:
        SSIM 值，float，范围 [-1, 1]，越接近 1 越好。
    """
    # 转为 [1, C, H, W] 以便做卷积
    pred   = pred.unsqueeze(0)    # [1, C, H, W]
    target = target.unsqueeze(0)  # [1, C, H, W]
    C = pred.shape[1]

    # 构建二维高斯核 [1, 1, window_size, window_size]，逐通道复制
    kernel_1d = _gaussian_kernel(window_size, sigma).to(pred.device)
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel    = kernel_2d.unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1)

    import torch.nn.functional as F
    pad = window_size // 2

    mu1 = F.conv2d(pred,   kernel, padding=pad, groups=C)
    mu2 = F.conv2d(target, kernel, padding=pad, groups=C)

    mu1_sq  = mu1 ** 2
    mu2_sq  = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred   * pred,   kernel, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(target * target, kernel, padding=pad, groups=C) - mu2_sq
    sigma12   = F.conv2d(pred   * target, kernel, padding=pad, groups=C) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean().item()


# =============================================================================
# 主评估函数
# =============================================================================

def evaluate(
    val_input_dir:  str,
    val_target_dir: str,
    ckpt_path:      str,
    img_size:       int = 256,
) -> None:
    """
    批量评估验证集，打印每张图的 PSNR/SSIM 并计算全集平均值。

    Args:
        val_input_dir  : 验证集退化图像目录
        val_target_dir : 验证集参考标准图目录
        ckpt_path      : 模型权重路径（best_model.pth）
        img_size       : 推理时 resize 的边长（与训练保持一致）
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Eval] 运行设备: {device}")

    # ── 加载模型 ──────────────────────────────────────────────────────────
    model = UNet(in_channels=3, out_channels=3).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[Eval] 已加载模型 ← Epoch {ckpt['epoch']}，"
          f"最优验证损失: {ckpt.get('best_val_loss', 'N/A')}")

    # ── 数据预处理（与 train.py 中的推理预处理保持完全一致）────────────────
    to_tensor_norm = T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # → [-1, 1]
    ])

    # ── 获取配对的验证集文件 ───────────────────────────────────────────────
    input_dir  = Path(val_input_dir)
    target_dir = Path(val_target_dir)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    input_files = sorted([f for f in input_dir.iterdir() if f.suffix.lower() in exts])

    psnr_list, ssim_list = [], []

    print(f"\n[Eval] 开始评估 {len(input_files)} 张验证图像...\n" + "─" * 55)

    for inp_path in tqdm(input_files, desc="Evaluating"):
        tgt_path = target_dir / inp_path.name
        if not tgt_path.exists():
            print(f"  [跳过] 找不到对应的参考图: {tgt_path.name}")
            continue

        # 读取图像
        inp_img = Image.open(inp_path).convert("RGB")
        tgt_img = Image.open(tgt_path).convert("RGB")

        # 预处理为张量 → [-1, 1]
        inp_tensor = to_tensor_norm(inp_img).unsqueeze(0).to(device)
        tgt_tensor = to_tensor_norm(tgt_img).unsqueeze(0).to(device)

        # 推理
        with torch.no_grad():
            pred_tensor = model(inp_tensor)  # [-1, 1]

        # 反归一化到 [0, 1] 后再计算指标
        pred_01   = ((pred_tensor.squeeze(0) + 1.0) / 2.0).clamp(0.0, 1.0)
        target_01 = ((tgt_tensor.squeeze(0)  + 1.0) / 2.0).clamp(0.0, 1.0)

        psnr = compute_psnr(pred_01, target_01)
        ssim = compute_ssim(pred_01, target_01)

        psnr_list.append(psnr)
        ssim_list.append(ssim)

    # ── 打印汇总结果 ──────────────────────────────────────────────────────
    if not psnr_list:
        print("[Eval] 警告：未能评估任何图像，请检查路径配置！")
        return

    avg_psnr = sum(psnr_list) / len(psnr_list)
    avg_ssim = sum(ssim_list) / len(ssim_list)

    print("\n" + "─" * 55)
    print(f"[Eval] 评估完成  共 {len(psnr_list)} 张图像")
    print(f"  平均 PSNR : {avg_psnr:.4f} dB")
    print(f"  平均 SSIM : {avg_ssim:.4f}")
    print("─" * 55)


# =============================================================================
# 命令行入口
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="U-Net 评估脚本 —— PSNR & SSIM")
    parser.add_argument("--val_input_dir",  default="data/val/input",
                        help="验证集退化图像目录")
    parser.add_argument("--val_target_dir", default="data/val/target",
                        help="验证集参考标准图目录")
    parser.add_argument("--ckpt",           default="checkpoints/best_model.pth",
                        help="模型权重路径")
    parser.add_argument("--img_size",       type=int, default=256,
                        help="图像推理时 resize 的边长（与训练保持一致）")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        val_input_dir  = args.val_input_dir,
        val_target_dir = args.val_target_dir,
        ckpt_path      = args.ckpt,
        img_size       = args.img_size,
    )
