"""
=============================================================================
模块 2: 网络拓扑结构定义 (Model Architecture) —— 标准 U-Net
论文: 基于深度卷积神经网络的水下偏色图像增强方法
=============================================================================

架构说明（严格对应论文 Table 1）：
    输入:  [B, 3,   256, 256]
    ─────────── Encoder ───────────
    enc1:  [B, 64,  256, 256]   Conv×2 → MaxPool
    enc2:  [B, 128, 128, 128]   Conv×2 → MaxPool
    enc3:  [B, 256,  64,  64]   Conv×2 → MaxPool
    enc4:  [B, 512,  32,  32]   Conv×2 → MaxPool
    ─────────── Bottleneck ────────
    bridge:[B,1024,  16,  16]   Conv×2  (无池化)
    ─────────── Decoder ───────────
    dec4:  [B, 512,  32,  32]   ConvTranspose + cat(enc4) → Conv×2
    dec3:  [B, 256,  64,  64]   ConvTranspose + cat(enc3) → Conv×2
    dec2:  [B, 128, 128, 128]   ConvTranspose + cat(enc2) → Conv×2
    dec1:  [B,  64, 256, 256]   ConvTranspose + cat(enc1) → Conv×2
    输出:  [B,   3, 256, 256]   1×1 Conv → Tanh
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 基本卷积块：Conv → BN → ReLU（重复 2 次）
# ---------------------------------------------------------------------------
class DoubleConv(nn.Module):
    """
    标准 U-Net 双卷积块:
        Conv(3×3, padding=1) → BatchNorm2d → ReLU
        Conv(3×3, padding=1) → BatchNorm2d → ReLU

    padding=1 保证特征图尺寸不变（same卷积）。

    Args:
        in_channels  : 输入通道数
        out_channels : 输出通道数
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            # 第一次卷积
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # 第二次卷积
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# Encoder 下采样块：DoubleConv → MaxPool(2×2)
# ---------------------------------------------------------------------------
class EncoderBlock(nn.Module):
    """
    收缩路径单层：
        特征提取 (DoubleConv) → 下采样 (MaxPool 2×2)

    Returns:
        skip  : 池化前的特征图（用于 Skip Connection）
        pooled: 池化后的特征图（传入下一层）
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        skip = self.conv(x)          # 保留给跳跃连接
        pooled = self.pool(skip)     # 下采样，传入下一层
        return skip, pooled


# ---------------------------------------------------------------------------
# Decoder 上采样块：ConvTranspose2d → cat(skip) → DoubleConv
# ---------------------------------------------------------------------------
class DecoderBlock(nn.Module):
    """
    扩张路径单层：
        转置卷积上采样 → 跳跃连接拼接 → DoubleConv

    Args:
        in_channels  : 上一层 decoder 的通道数
        out_channels : 本层输出通道数
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        # 转置卷积：通道数减半，空间尺寸加倍
        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        # 拼接后通道数 = out_channels（来自上采样）+ out_channels（来自 skip）
        self.conv = DoubleConv(out_channels * 2, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)

        # ── 尺寸对齐安全处理 ──────────────────────────────────────────
        # 理论上 256 可被 2^4=16 整除，不需要裁剪；
        # 但对于非 256×256 输入（如奇数尺寸），做 Center Crop 防止维度不匹配。
        if x.shape != skip.shape:
            diff_h = skip.shape[2] - x.shape[2]
            diff_w = skip.shape[3] - x.shape[3]
            x = F.pad(x, [
                diff_w // 2, diff_w - diff_w // 2,
                diff_h // 2, diff_h - diff_h // 2,
            ])

        # Skip Connection：在通道维度拼接 Encoder 特征图
        x = torch.cat([skip, x], dim=1)   # [B, out_ch*2, H, W]
        return self.conv(x)


# ---------------------------------------------------------------------------
# 完整 U-Net 模型
# ---------------------------------------------------------------------------
class UNet(nn.Module):
    """
    标准 U-Net 用于水下图像增强。

    编码器通道序列: 3 → 64 → 128 → 256 → 512 → 1024 (bottleneck)
    解码器通道序列: 1024 → 512 → 256 → 128 → 64 → 3 (output)

    Args:
        in_channels  : 输入图像通道数（RGB=3）
        out_channels : 输出图像通道数（RGB=3）
        base_ch      : 第一层特征通道数（默认 64）
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_ch: int = 64,
    ) -> None:
        super().__init__()

        # ── Encoder（收缩路径）————————————————————————————————
        # 每层结束后空间尺寸减半，通道数加倍
        self.enc1 = EncoderBlock(in_channels, base_ch)       # 3   → 64
        self.enc2 = EncoderBlock(base_ch,     base_ch * 2)   # 64  → 128
        self.enc3 = EncoderBlock(base_ch * 2, base_ch * 4)   # 128 → 256
        self.enc4 = EncoderBlock(base_ch * 4, base_ch * 8)   # 256 → 512

        # ── Bottleneck（网络瓶颈最深层）—————————————————————————
        # 不做 MaxPool，保留最小分辨率 (256/16=16)
        self.bottleneck = DoubleConv(base_ch * 8, base_ch * 16)  # 512 → 1024

        # ── Decoder（扩张路径）————————————————————————————————
        # 每层上采样，并接收对应 Encoder 层的 skip connection
        self.dec4 = DecoderBlock(base_ch * 16, base_ch * 8)  # 1024 → 512
        self.dec3 = DecoderBlock(base_ch * 8,  base_ch * 4)  # 512  → 256
        self.dec2 = DecoderBlock(base_ch * 4,  base_ch * 2)  # 256  → 128
        self.dec1 = DecoderBlock(base_ch * 2,  base_ch)      # 128  → 64

        # ── Output Head ———————————————————————————————————
        # 1×1 卷积将 64 通道映射到输出通道数 (3)
        # Tanh 将像素值映射到 [-1, 1]，与归一化方式匹配
        self.output_conv = nn.Sequential(
            nn.Conv2d(base_ch, out_channels, kernel_size=1),
            nn.Tanh(),
        )

        # 权重初始化（Kaiming Normal 适合 ReLU 激活）
        self._init_weights()

    def _init_weights(self) -> None:
        """使用 Kaiming Normal 初始化所有卷积层权重。"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            x: FloatTensor [B, 3, 256, 256]，归一化后的退化图

        Returns:
            FloatTensor [B, 3, 256, 256]，增强后的干净图（范围 [-1,1]）
        """
        # ── Encoder 前向 ──────────────────────────────────────
        skip1, x = self.enc1(x)    # skip1:[B,  64,256,256]  x:[B,  64,128,128]
        skip2, x = self.enc2(x)    # skip2:[B, 128,128,128]  x:[B, 128, 64, 64]
        skip3, x = self.enc3(x)    # skip3:[B, 256, 64, 64]  x:[B, 256, 32, 32]
        skip4, x = self.enc4(x)    # skip4:[B, 512, 32, 32]  x:[B, 512, 16, 16]

        # ── Bottleneck ────────────────────────────────────────
        x = self.bottleneck(x)     # [B, 1024, 16, 16]

        # ── Decoder 前向（每层传入对应 skip）────────────────────
        x = self.dec4(x, skip4)    # [B, 512, 32,  32]
        x = self.dec3(x, skip3)    # [B, 256, 64,  64]
        x = self.dec2(x, skip2)    # [B, 128, 128, 128]
        x = self.dec1(x, skip1)    # [B,  64, 256, 256]

        # ── Output ────────────────────────────────────────────
        return self.output_conv(x) # [B,   3, 256, 256]


# ---------------------------------------------------------------------------
# 快速自测
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(in_channels=3, out_channels=3).to(device)

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] 总参数量     : {total_params:,}")
    print(f"[Model] 可训练参数量 : {trainable_params:,}")

    # 验证前向传播尺寸
    dummy_input = torch.randn(2, 3, 256, 256).to(device)
    with torch.no_grad():
        dummy_output = model(dummy_input)
    print(f"[Model] 输入尺寸     : {dummy_input.shape}")   # [2, 3, 256, 256]
    print(f"[Model] 输出尺寸     : {dummy_output.shape}")  # [2, 3, 256, 256]
    print(f"[Model] 输出值范围   : [{dummy_output.min():.3f}, {dummy_output.max():.3f}]")
    print("[Model] 网络结构验证通过 ✓")
