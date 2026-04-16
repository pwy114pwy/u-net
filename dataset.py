"""
=============================================================================
模块 1: 数据集与数据加载模块 (Dataset & DataLoader)
论文: 基于深度卷积神经网络的水下偏色图像增强方法
=============================================================================

目录组织约定（请按此结构准备数据）：
    data/
    ├── train/
    │   ├── input/      # 退化（偏色）水下图像
    │   └── target/     # 对应的参考（干净）图像
    └── val/
        ├── input/
        └── target/

输入/输出图像的文件名必须一一对应（例如 001.png 对应 001.png）。
=============================================================================
"""

import os
from pathlib import Path    # 用于更现代、简洁地处理文件路径
from PIL import Image   # Python 图像处理标准库，用于读取图片

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


# ---------------------------------------------------------------------------
# 自定义数据集类
# ---------------------------------------------------------------------------
class UnderwaterDataset(Dataset):
    """
    水下偏色图像增强 配对数据集。

    Args:
        input_dir  (str | Path): 退化图像所在目录
        target_dir (str | Path): 参考干净图像所在目录
        img_size   (int)       : 统一缩放的边长，论文使用 256
        augment    (bool)      : 是否开启训练期间的随机数据增广
    """

    # 支持的图像后缀
    IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    def __init__(
        self,
        input_dir: str,
        target_dir: str,
        img_size: int = 256,
        augment: bool = False,
    ) -> None:
        super().__init__()

        self.input_dir = Path(input_dir)
        self.target_dir = Path(target_dir)
        self.img_size = img_size
        self.augment = augment

        # 收集所有合法图像文件（按文件名排序保证顺序一致）
        self.input_files = sorted(
            [
                p
                for p in self.input_dir.iterdir()
                if p.suffix.lower() in self.IMG_EXTENSIONS
            ]
        )

        # 校验：input 与 target 数量应严格一致
        assert len(self.input_files) > 0, (
            f"在 {self.input_dir} 中未找到任何图像文件，"
            "请检查路径或图像格式。"
        )

        target_names = {
            p.name
            for p in self.target_dir.iterdir()
            if p.suffix.lower() in self.IMG_EXTENSIONS
        }
        for f in self.input_files:
            assert f.name in target_names, (
                f"退化图 {f.name} 在 target 目录中找不到对应文件，"
                "请确保输入与参考图像一一对应。"
            )

        # ----------------------------------------------------------------
        # 预处理 Pipeline
        # ----------------------------------------------------------------
        # 共享变换：resize → tensor → 归一化到 [-1, 1]
        # 使用 [-1,1] 归一化（均值=0.5，标准差=0.5）更有利于 tanh 激活稳定训练；
        # 若最终输出层使用 sigmoid，改为 [0,1] 即去掉 Normalize 也可。
        shared_transforms = [
            # BICUBIC（双三次插值）是一种高质量的缩放算法，相比普通缩放，它能更好地保留水下图像的边缘细节
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
            
            T.ToTensor(),                          # PIL [0,255] → Tensor [0.0, 1.0]
            # 将[0, 1]的范围进一步拉伸到[-1, 1]
            T.Normalize(mean=[0.5, 0.5, 0.5],      # [0,1] → [-1, 1]
                        std=[0.5, 0.5, 0.5]),
        ]
        self.transform = T.Compose(shared_transforms)

        # 数据增广（仅训练期间使用）：水平翻转、垂直翻转
        # 注意：增广必须对 input 和 target 同步施加 —— 这里使用随机种子同步
        self.augment_transforms = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
        ])

    def __len__(self) -> int:
        return len(self.input_files)

    def __getitem__(self, index: int) -> dict:
        """
        Returns:
            dict 包含：
                "input"  : FloatTensor [3, H, W], 归一化后的退化图
                "target" : FloatTensor [3, H, W], 归一化后的参考图
                "name"   : str, 文件名（便于推理时保存结果）
        """
        input_path = self.input_files[index]
        target_path = self.target_dir / input_path.name

        # 以 RGB 模式读取（忽略 Alpha 通道）
        # convert("RGB") 非常重要，它能把灰度
        # 图或带透明通道的 PNG 统一转成标准的红绿蓝
        #  3 通道图，避免模型报错
        input_img = Image.open(input_path).convert("RGB")
        target_img = Image.open(target_path).convert("RGB")

        if self.augment:
            # 使用相同的随机种子确保 input 与 target 施加相同的几何变换
            seed = torch.randint(0, 2**32, (1,)).item()
            torch.manual_seed(seed)
            input_img = self.augment_transforms(input_img)
            torch.manual_seed(seed)
            target_img = self.augment_transforms(target_img)

        input_tensor = self.transform(input_img)
        target_tensor = self.transform(target_img)

        return {
            "input": input_tensor,
            "target": target_tensor,
            "name": input_path.name,
        }


# ---------------------------------------------------------------------------
# DataLoader 工厂函数
# ---------------------------------------------------------------------------
def build_dataloader(
    input_dir: str,
    target_dir: str,
    batch_size: int = 8,
    img_size: int = 256,
    augment: bool = False,
    num_workers: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    """
    构建水下图像数据加载器。

    Args:
        input_dir   : 退化图像目录
        target_dir  : 参考图像目录
        batch_size  : 每批次样本数
        img_size    : 图像缩放尺寸（默认 256）
        augment     : 是否开启增广
        num_workers : 数据加载并行进程数（Windows 建议设为 0 或 2）
        shuffle     : 是否随机打乱（验证集应设为 False）

    Returns:
        DataLoader 对象
    """
    dataset = UnderwaterDataset(
        input_dir=input_dir,
        target_dir=target_dir,
        img_size=img_size,
        augment=augment,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,       # 加速 CPU→GPU 数据传输
        # 丢弃余数
        drop_last=False,
    )
    return loader


# ---------------------------------------------------------------------------
# 快速自测：直接运行该文件时执行
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    # 请修改为你的实际数据路径
    TRAIN_INPUT = "data/train/input"
    TRAIN_TARGET = "data/train/target"

    if not Path(TRAIN_INPUT).exists():
        print("[Warning] 数据目录不存在，跳过 DataLoader 测试。")
        sys.exit(0)

    loader = build_dataloader(
        input_dir=TRAIN_INPUT,
        target_dir=TRAIN_TARGET,
        batch_size=4,
        augment=True,
        num_workers=0,   # Windows 调试时建议设为 0
    )
    batch = next(iter(loader))
    print(f"[Dataset] 样本总数  : {len(loader.dataset)}")
    print(f"[Dataset] input  shape: {batch['input'].shape}")   # [B, 3, 256, 256]
    print(f"[Dataset] target shape: {batch['target'].shape}")  # [B, 3, 256, 256]
    print(f"[Dataset] 像素值范围   : [{batch['input'].min():.2f}, {batch['input'].max():.2f}]")
