"""
=============================================================================
模块 4: 训练循环与推理验证 (Training & Inference)
论文: 基于深度卷积神经网络的水下偏色图像增强方法
=============================================================================

使用方法：
    训练: python train.py
    推理: python train.py --mode infer --input_img path/to/underwater.jpg
          （或直接调用 infer_single() 函数）

超参数（论文设定）：
    优化器    : Adam
    学习率    : 0.001
    Batch Size: 8（可根据显存调整）
    输入尺寸  : 256×256
    损失函数  : L_MSE + L_SSIM
=============================================================================
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast   # AMP 混合精度
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.utils import save_image
from PIL import Image
import torchvision.transforms as T

# 导入本项目其他模块
from dataset import build_dataloader
from model import UNet
from loss import HybridLoss


# =============================================================================
# 工具函数
# =============================================================================

def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    将 [-1, 1] 范围的张量反归一化回 [0, 1]，便于保存为图像。

    Args:
        tensor: FloatTensor，值域 [-1, 1]

    Returns:
        FloatTensor，值域 [0, 1]（已自动 clamp）
    """
    return ((tensor + 1.0) / 2.0).clamp(0.0, 1.0)


def save_checkpoint(
    state: dict,
    filepath: str,
    is_best: bool,
    best_path: str,
) -> None:
    """
    保存训练 Checkpoint。

    Args:
        state     : 包含 epoch、model_state_dict、optimizer_state_dict、loss 等
        filepath  : 当前 epoch 的保存路径（如 checkpoints/epoch_010.pth）
        is_best   : 是否为当前最优模型
        best_path : 最优模型的固定保存路径（如 checkpoints/best_model.pth）
    """
    torch.save(state, filepath)
    if is_best:
        import shutil
        shutil.copyfile(filepath, best_path)
        print(f"  ★ 最优模型已更新 → {best_path}")


# =============================================================================
# 训练主循环
# =============================================================================

def train(config: dict) -> None:
    """
    完整训练流程：初始化 → 数据加载 → 训练循环 → 保存最优模型。

    Args:
        config: 超参数字典，包含所有训练配置
    """
    # ── 环境准备 ──────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] 运行设备: {device}")

    # checkpoint 保存目录
    ckpt_dir = Path(config["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = str(ckpt_dir / "best_model.pth")

    # 可视化样本保存目录（训练中对比效果）
    vis_dir = Path(config["vis_dir"])
    vis_dir.mkdir(parents=True, exist_ok=True)

    # ── 数据加载 ──────────────────────────────────────────────────────────
    print("[Train] 构建数据加载器...")
    train_loader = build_dataloader(
        input_dir=config["train_input_dir"],
        target_dir=config["train_target_dir"],
        batch_size=config["batch_size"],
        img_size=config["img_size"],
        augment=True,
        num_workers=config["num_workers"],
        shuffle=True,
    )
    val_loader = build_dataloader(
        input_dir=config["val_input_dir"],
        target_dir=config["val_target_dir"],
        batch_size=config["batch_size"],
        img_size=config["img_size"],
        augment=False,
        num_workers=config["num_workers"],
        shuffle=False,
    )
    print(f"[Train] 训练集: {len(train_loader.dataset)} 张  "
          f"| 验证集: {len(val_loader.dataset)} 张")

    # ── 模型、损失、优化器 ────────────────────────────────────────────────
    model = UNet(in_channels=3, out_channels=3).to(device)

    # 若指定了预训练权重则加载（断点续训）
    start_epoch = 1
    best_val_loss = float("inf")
    if config.get("resume"):
        ckpt = torch.load(config["resume"], map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"[Train] 从 Epoch {ckpt['epoch']} 恢复训练，最优验证损失: {best_val_loss:.6f}")

    # 论文指定: Adam 优化器, lr=0.001
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    # 学习率调度器：验证 Loss 连续 5 个 epoch 不下降则将 lr 缩减一半
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    # 混合损失函数
    criterion = HybridLoss(
        ssim_weight=config["ssim_weight"],
        data_range=2.0,   # 归一化范围 [-1,1]
    ).to(device)

    # -- AMP 初始化 -------------------------------------------------------
    # 仅在 CUDA 设备上启用 AMP；CPU 训练时 enabled=False 退化为普通 float32，
    # 不会引发任何兼容性问题。
    use_amp = (device.type == "cuda")
    scaler  = GradScaler(enabled=use_amp)
    if use_amp:
        print("[Train] AMP 混合精度已启用 (float16 前向 + float32 参数更新)")
    else:
        print("[Train] CPU 模式：AMP 已跳过，使用标准 float32 训练")

    # ── 训练循环 ──────────────────────────────────────────────────────────
    print(f"\n[Train] 开始训练，共 {config['max_epochs']} 个 Epoch\n" + "─" * 60)

    for epoch in range(start_epoch, config["max_epochs"] + 1):
        t0 = time.time()

        # ── 训练阶段 ──────────────────────────────────────────────────────
        model.train()
        train_loss_sum = 0.0

        for batch in train_loader:
            inputs  = batch["input"].to(device)    # [B, 3, 256, 256]
            targets = batch["target"].to(device)   # [B, 3, 256, 256]

            optimizer.zero_grad()

            # -- AMP 前向传播（float16 计算，自动降低显存）------------------
            with autocast(enabled=use_amp):
                preds = model(inputs)              # [B, 3, 256, 256]
                loss, loss_dict = criterion(preds, targets)

            # -- AMP 反向传播 -----------------------------------------------
            # scaler.scale() 将 loss 放大以防止 float16 梯度下溢
            scaler.scale(loss).backward()

            # unscale_ 必须在 clip_grad_norm_ 之前调用，
            # 否则裁剪的是"放大后"的梯度，max_norm 语义会失真
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # scaler.step() 内部自动检测梯度是否为 inf/nan：
            #   若正常 -> 等价于 optimizer.step()
            #   若异常 -> 跳过本步更新（不会污染模型权重）
            scaler.step(optimizer)
            scaler.update()   # 动态调整 loss 放大系数

            train_loss_sum += loss.item()

        avg_train_loss = train_loss_sum / len(train_loader)

        # ── 验证阶段 ──────────────────────────────────────────────────────
        model.eval()
        val_loss_sum = 0.0

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                inputs  = batch["input"].to(device)
                targets = batch["target"].to(device)

                preds = model(inputs)
                val_loss, _ = criterion(preds, targets)
                val_loss_sum += val_loss.item()

                # 保存第一批次的可视化对比图（每隔 N 个 epoch）
                if i == 0 and epoch % config["vis_every"] == 0:
                    # 拼接：[退化输入 | 模型输出 | 参考干净图]
                    n_show = min(4, inputs.size(0))
                    vis = torch.cat([
                        denormalize(inputs[:n_show]),
                        denormalize(preds[:n_show]),
                        denormalize(targets[:n_show]),
                    ], dim=0)
                    save_image(vis, vis_dir / f"epoch_{epoch:04d}.png", nrow=n_show)

        avg_val_loss = val_loss_sum / len(val_loader)

        # 学习率自适应调整
        scheduler.step(avg_val_loss)

        # 当前 epoch 耗时
        elapsed = time.time() - t0

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch:4d}/{config['max_epochs']}]  "
            f"Train Loss: {avg_train_loss:.6f}  "
            f"Val Loss: {avg_val_loss:.6f}  "
            f"LR: {current_lr:.2e}  "
            f"Time: {elapsed:.1f}s"
        )

        # ── 保存 Checkpoint ────────────────────────────────────────────────
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss

        state = {
            "epoch":            epoch,
            "model_state_dict": model.state_dict(),
            "optim_state_dict": optimizer.state_dict(),
            "train_loss":       avg_train_loss,
            "val_loss":         avg_val_loss,
            "best_val_loss":    best_val_loss,
            "config":           config,
        }

        # 每隔 save_every 个 epoch 保存一次常规 checkpoint
        if epoch % config["save_every"] == 0:
            save_checkpoint(
                state,
                filepath=str(ckpt_dir / f"epoch_{epoch:04d}.pth"),
                is_best=is_best,
                best_path=best_ckpt_path,
            )
        elif is_best:
            # 即使不在保存间隔内，只要是最优模型也保存
            save_checkpoint(
                state,
                filepath=str(ckpt_dir / f"epoch_{epoch:04d}_best.pth"),
                is_best=True,
                best_path=best_ckpt_path,
            )

    print("\n" + "─" * 60)
    print(f"[Train] 训练完成！最优验证损失: {best_val_loss:.6f}")
    print(f"[Train] 最优模型已保存至: {best_ckpt_path}")


# =============================================================================
# 单张图像推理函数
# =============================================================================

def infer_single(
    input_img_path: str,
    checkpoint_path: str,
    output_path: str = "output_enhanced.png",
    img_size: int = 256,
) -> None:
    """
    对单张水下偏色图像执行推理，并保存增强后的结果。

    Args:
        input_img_path  : 输入退化图像路径（支持 jpg/png 等格式）
        checkpoint_path : 模型权重文件路径（best_model.pth）
        output_path     : 增强结果的保存路径
        img_size        : 模型处理的图像尺寸（默认 256，与训练一致）
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 加载模型 ──────────────────────────────────────────────────────────
    model = UNet(in_channels=3, out_channels=3).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[Infer] 已加载模型 (训练至 Epoch {ckpt['epoch']}，"
          f"最优验证损失: {ckpt.get('best_val_loss', 'N/A')})")

    # ── 图像预处理（与训练时保持完全一致）──────────────────────────────────
    transform = T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # → [-1, 1]
    ])

    # 记录原始图像尺寸（推理后恢复原始分辨率）
    raw_img = Image.open(input_img_path).convert("RGB")
    orig_size = raw_img.size  # (W, H)，PIL 格式

    input_tensor = transform(raw_img).unsqueeze(0).to(device)  # [1, 3, 256, 256]

    # ── 前向推理 ──────────────────────────────────────────────────────────
    with torch.no_grad():
        output_tensor = model(input_tensor)  # [1, 3, 256, 256]，范围 [-1, 1]

    # ── 后处理：反归一化 → 恢复原始分辨率 → 保存 ────────────────────────────
    output_tensor = denormalize(output_tensor)  # [1, 3, 256, 256]，范围 [0, 1]

    # 将结果恢复到原始分辨率（避免输出总是 256×256 的小图）
    output_img = T.ToPILImage()(output_tensor.squeeze(0).cpu())
    output_img = output_img.resize(orig_size, Image.BICUBIC)
    output_img.save(output_path)

    print(f"[Infer] 增强完成！原始尺寸: {orig_size[0]}×{orig_size[1]}")
    print(f"[Infer] 结果已保存至: {output_path}")


# =============================================================================
# 命令行入口
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="水下偏色图像增强 U-Net —— 训练 & 推理"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "infer"],
        help="运行模式: train（训练）或 infer（推理）",
    )

    # ── 训练参数 ──
    parser.add_argument("--train_input_dir",  default="data/train/input")
    parser.add_argument("--train_target_dir", default="data/train/target")
    parser.add_argument("--val_input_dir",    default="data/val/input")
    parser.add_argument("--val_target_dir",   default="data/val/target")
    parser.add_argument("--checkpoint_dir",   default="checkpoints")
    parser.add_argument("--vis_dir",          default="vis_results")
    parser.add_argument("--max_epochs",  type=int,   default=200,  help="最大训练轮次")
    parser.add_argument("--batch_size",  type=int,   default=8,    help="批次大小")
    parser.add_argument("--img_size",    type=int,   default=256,  help="图像边长")
    parser.add_argument("--lr",          type=float, default=1e-3, help="初始学习率")
    parser.add_argument("--ssim_weight", type=float, default=1.0,  help="SSIM 损失权重")
    parser.add_argument("--num_workers", type=int,   default=4,    help="DataLoader 进程数")
    parser.add_argument("--save_every",  type=int,   default=10,   help="每 N epoch 保存一次")
    parser.add_argument("--vis_every",   type=int,   default=5,    help="每 N epoch 保存可视化")
    parser.add_argument("--resume",      type=str,   default="",   help="断点续训的 checkpoint 路径")

    # ── 推理参数 ──
    parser.add_argument("--input_img",   default="test_input.jpg",   help="待增强的水下图像路径")
    parser.add_argument("--output_img",  default="output_enhanced.png", help="增强结果保存路径")
    parser.add_argument("--ckpt",        default="checkpoints/best_model.pth", help="推理使用的模型权重")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "train":
        # ── 训练配置字典 ──────────────────────────────────────────────────
        config = {
            "train_input_dir":  args.train_input_dir,
            "train_target_dir": args.train_target_dir,
            "val_input_dir":    args.val_input_dir,
            "val_target_dir":   args.val_target_dir,
            "checkpoint_dir":   args.checkpoint_dir,
            "vis_dir":          args.vis_dir,
            "max_epochs":       args.max_epochs,
            "batch_size":       args.batch_size,
            "img_size":         args.img_size,
            "lr":               args.lr,              # 0.001（论文）
            "ssim_weight":      args.ssim_weight,
            "num_workers":      args.num_workers,
            "save_every":       args.save_every,
            "vis_every":        args.vis_every,
            "resume":           args.resume or None,
        }
        train(config)

    elif args.mode == "infer":
        infer_single(
            input_img_path=args.input_img,
            checkpoint_path=args.ckpt,
            output_path=args.output_img,
            img_size=args.img_size,
        )
