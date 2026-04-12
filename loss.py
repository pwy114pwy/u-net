"""
=============================================================================
模块 3: 自定义混合损失函数 (Hybrid Loss Function)
论文: 基于深度卷积神经网络的水下偏色图像增强方法
=============================================================================

论文提出的混合损失函数：
    L = L_MSE + L_SSIM

    L_MSE  = MSE(pred, target)
    L_SSIM = 1 - SSIM(pred, target)
           （SSIM 越高越好，取 1-SSIM 转为最小化问题）

SSIM 的实现：
    - 使用可微分的 2D 高斯卷积核逐通道计算亮度(luminance)、
      对比度(contrast)、结构(structure)三个相似度分量。
    - 公式参考 Wang et al. (2004) "Image Quality Assessment..."
    - 常数 C1=(0.01*L)^2, C2=(0.03*L)^2，其中 L=2（[-1,1]值域范围）

注意：此处 SSIM 完全使用 PyTorch 原生算子实现，无需额外第三方库，
全程可微分，支持反向传播。
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 高斯核生成（SSIM 核心组件）
# ---------------------------------------------------------------------------
def _gaussian_kernel(kernel_size: int, sigma: float, channels: int) -> torch.Tensor:
    """
    生成 2D 高斯卷积核，供 SSIM 计算均值与方差使用。

    Args:
        kernel_size : 卷积核大小（论文通常取 11）
        sigma       : 高斯标准差（通常取 1.5）
        channels    : 图像通道数（RGB=3）

    Returns:
        Tensor [channels, 1, kernel_size, kernel_size]
        每个通道独立一个相同的高斯核（分组卷积使用）
    """
    # 生成 1D 高斯分布
    coords = torch.arange(kernel_size, dtype=torch.float32)
    coords -= kernel_size // 2
    g_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g_1d /= g_1d.sum()

    # 外积得到 2D 高斯核，并归一化
    g_2d = g_1d.unsqueeze(1) @ g_1d.unsqueeze(0)    # [k, k]
    g_2d = g_2d / g_2d.sum()

    # 扩展为 [channels, 1, k, k]（用于分组深度卷积，逐通道计算）
    kernel = g_2d.unsqueeze(0).unsqueeze(0)          # [1, 1, k, k]
    kernel = kernel.expand(channels, 1, kernel_size, kernel_size).contiguous()
    return kernel


# ---------------------------------------------------------------------------
# 可微分 SSIM 计算（核心实现）
# ---------------------------------------------------------------------------
class SSIM(nn.Module):
    """
    可微分的结构相似性 (SSIM) 计算模块。

    SSIM 公式:
        SSIM(x, y) = (2μ_x μ_y + C1)(2σ_xy + C2)
                     ─────────────────────────────
                     (μ_x² + μ_y² + C1)(σ_x² + σ_y² + C2)

    其中:
        μ_x, μ_y   : 局部均值（高斯加权）
        σ_x², σ_y² : 局部方差
        σ_xy       : 局部协方差
        C1, C2     : 稳定常数，防止分母为零

    Args:
        kernel_size : 高斯窗口大小（默认 11，论文标准）
        sigma       : 高斯核标准差（默认 1.5）
        data_range  : 像素值范围（[-1,1] 则为 2.0；[0,1] 则为 1.0）
        channels    : 图像通道数（默认 3，RGB）
    """

    def __init__(
        self,
        kernel_size: int = 11,
        sigma: float = 1.5,
        data_range: float = 2.0,   # 归一化为 [-1,1] 时范围为 2
        channels: int = 3,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.channels = channels
        self.padding = kernel_size // 2

        # 稳定常数（不参与梯度更新）
        self.C1 = (0.01 * data_range) ** 2
        self.C2 = (0.03 * data_range) ** 2

        # 注册高斯核为 Buffer（随模型 .to(device) 自动迁移，但不参与优化）
        kernel = _gaussian_kernel(kernel_size, sigma, channels)
        self.register_buffer("kernel", kernel)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算批次平均 SSIM 值。

        Args:
            pred   : 模型预测图 [B, C, H, W]
            target : 参考干净图 [B, C, H, W]

        Returns:
            标量 Tensor，批次内所有图像所有通道的平均 SSIM
        """
        # 分组深度卷积：每个通道独立使用同一高斯核
        kwargs = dict(
            weight=self.kernel,
            padding=self.padding,
            groups=self.channels,
        )

        # 局部均值（高斯加权均值）
        mu_pred   = F.conv2d(pred,   **kwargs)
        mu_target = F.conv2d(target, **kwargs)

        mu_pred_sq   = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_pred_target = mu_pred * mu_target

        # 局部方差与协方差
        # Var(X) = E[X²] - E[X]²
        sigma_pred_sq   = F.conv2d(pred * pred,     **kwargs) - mu_pred_sq
        sigma_target_sq = F.conv2d(target * target, **kwargs) - mu_target_sq
        sigma_pred_target = F.conv2d(pred * target, **kwargs) - mu_pred_target

        # SSIM 分子与分母
        numerator   = (2 * mu_pred_target + self.C1) * (2 * sigma_pred_target + self.C2)
        denominator = (mu_pred_sq + mu_target_sq + self.C1) * (sigma_pred_sq + sigma_target_sq + self.C2)

        ssim_map = numerator / denominator   # [B, C, H', W']，逐像素 SSIM

        # 对所有空间位置、通道、批次取均值
        return ssim_map.mean()


# ---------------------------------------------------------------------------
# 混合损失函数：L = L_MSE + L_SSIM
# ---------------------------------------------------------------------------
class HybridLoss(nn.Module):
    """
    论文定义的混合损失：L = L_MSE + λ * L_SSIM

    L_MSE  = F.mse_loss(pred, target)
    L_SSIM = 1 - SSIM(pred, target)

    Args:
        ssim_weight : SSIM 损失的权重 λ（默认 1.0，与论文一致）
        kernel_size : SSIM 高斯核大小
        sigma       : SSIM 高斯核标准差
        data_range  : 像素值范围（[-1,1] 归一化 → 2.0）
    """

    def __init__(
        self,
        ssim_weight: float = 1.0,
        kernel_size: int = 11,
        sigma: float = 1.5,
        data_range: float = 2.0,
    ) -> None:
        super().__init__()
        self.ssim_weight = ssim_weight
        self.ssim_module = SSIM(
            kernel_size=kernel_size,
            sigma=sigma,
            data_range=data_range,
            channels=3,
        )

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        计算混合损失。

        Args:
            pred   : 模型输出 [B, 3, H, W]
            target : 参考干净图 [B, 3, H, W]

        Returns:
            total_loss : 标量 Tensor，总损失值（用于反向传播）
            loss_dict  : 各分项损失的字典（用于日志记录）
        """
        # L_MSE：均方误差
        l_mse = F.mse_loss(pred, target)

        # L_SSIM：结构相似性损失（1 - SSIM 将最大化问题转为最小化问题）
        l_ssim = 1.0 - self.ssim_module(pred, target)

        # 混合损失（论文公式 λ=1）
        total_loss = l_mse + self.ssim_weight * l_ssim

        loss_dict = {
            "loss_total": total_loss.item(),
            "loss_mse":   l_mse.item(),
            "loss_ssim":  l_ssim.item(),
        }
        return total_loss, loss_dict


# ---------------------------------------------------------------------------
# 快速自测
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    criterion = HybridLoss(data_range=2.0).to(device)

    # 模拟一批预测与标签（范围 [-1, 1]）
    pred   = torch.randn(2, 3, 256, 256).to(device)
    target = torch.randn(2, 3, 256, 256).to(device)

    loss, info = criterion(pred, target)
    print(f"[Loss] 总损失     : {info['loss_total']:.6f}")
    print(f"[Loss]   L_MSE   : {info['loss_mse']:.6f}")
    print(f"[Loss]   L_SSIM  : {info['loss_ssim']:.6f}")

    # 验证可微分性
    loss.backward()
    print("[Loss] 反向传播验证通过 ✓")

    # 完全相同的 pred 和 target，SSIM=1，L_SSIM 应趋近 0
    pred_same = target.clone()
    loss2, info2 = criterion(pred_same, target)
    print(f"\n[Loss] 完全相同时的 L_SSIM : {info2['loss_ssim']:.8f} (应趋近 0)")
