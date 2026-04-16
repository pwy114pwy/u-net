# 水下偏色图像增强 U-Net 复现工程

> 论文：《基于深度卷积神经网络的水下偏色图像增强方法》
> 框架：PyTorch

---

## 项目结构

```
u-net/
├── dataset.py        # 模块 1：数据集与数据加载
├── model.py          # 模块 2：U-Net 网络结构
├── loss.py           # 模块 3：MSE + SSIM 混合损失函数
├── train.py          # 模块 4：训练循环 & 推理验证
├── requirements.txt  # 依赖列表
└── data/             # 数据目录（需自行准备）
    ├── train/
    │   ├── input/    # 退化（偏色）水下图像
    │   └── target/   # 对应参考（干净）图像
    └── val/
        ├── input/
        └── target/
```

---

## 环境安装

```bash
pip install -r requirements.txt
```

---

## 快速开始

### 1. 准备数据

将配对图像按上述结构放置，**input/ 与 target/ 下的文件名必须一一对应**。

### 2. 启动训练

```bash
python train.py --mode train \
    --train_input_dir data/train/input \
    --train_target_dir data/train/target \
    --val_input_dir data/val/input \
    --val_target_dir data/val/target \
    --max_epochs 200 \
    --batch_size 8 \
    --lr 0.001
```

**Windows 注意**：将 `--num_workers` 设为 `0` 或 `2` 以避免多进程问题：
```bash
python train.py --mode train --num_workers 0 ...
```

### 3. 断点续训

```bash
python train.py --mode train --resume checkpoints/epoch_0100.pth
```

### 4. 单张图像推理

```bash
python train.py --mode infer \
    --input_img path/to/underwater_image.jpg \
    --output_img output_enhanced.png \
    --ckpt checkpoints/best_model.pth
```

---

## 模块技术说明

### 模块 1 —— Dataset（`dataset.py`）

| 项目 | 详情 |
|------|------|
| 输入 | RGB 图像（JPG / PNG / BMP / TIF） |
| 输出尺寸 | 256 × 256 × 3 |
| 归一化 | `[-1, 1]`（均值 0.5，标准差 0.5） |
| 数据增广 | 随机水平翻转 + 随机垂直翻转（同步施加于 input/target） |

### 模块 2 —— U-Net（`model.py`）

| 路径 | 层数 | 通道数序列 |
|------|------|-----------|
| Encoder（收缩路径） | 4 | 3→64→128→256→512 |
| Bottleneck | 1 | 512→1024 |
| Decoder（扩张路径） | 4 | 1024→512→256→128→64 |
| Output Head | — | 64→3，Tanh 激活 |

- 下采样：MaxPool 2×2
- 上采样：转置卷积（ConvTranspose2d，步长 2）
- Skip Connection：`torch.cat()` 在通道维度拼接对应 Encoder 特征图
- 参数量：约 **31M**

### 模块 3 —— 混合损失函数（`loss.py`）

$$L = L_{MSE} + \lambda \cdot L_{SSIM}$$

$$L_{SSIM} = 1 - \text{SSIM}(\hat{y}, y)$$

- SSIM 使用 11×11 高斯核（σ=1.5），纯 PyTorch 实现，全程可微分
- 数据范围 `data_range=2.0`（对应归一化范围 `[-1,1]`）
- 默认 λ=1.0

### 模块 4 —— 训练循环（`train.py`）

| 超参数 | 值 |
|--------|----|
| 优化器 | Adam |
| 初始学习率 | 0.001 |
| LR 调度 | ReduceLROnPlateau（patience=5，factor=0.5） |
| 梯度裁剪 | max_norm=1.0 |
| 最优 Checkpoint | Val Loss 最低时自动保存至 `checkpoints/best_model.pth` |
| **混合精度训练** | **AMP（自动混合精度），GPU 上默认启用** |

#### 训练优化：AMP 自动混合精度（Automatic Mixed Precision）

在 GPU 设备上训练时，程序会自动启用 `torch.cuda.amp`，将前向传播与损失计算切换到 **float16** 精度，参数更新仍保持 **float32**，从而在不损失模型精度的前提下获得显著加速。

| 项目 | 说明 |
|------|------|
| 前向传播精度 | float16（`autocast` 自动管理） |
| 参数更新精度 | float32（GradScaler 自动还原） |
| 梯度裁剪 | `scaler.unscale_()` 后执行，语义正确 |
| 异常保护 | 检测到 `inf`/`nan` 梯度时自动跳过本步更新 |
| CPU 兼容 | CPU 训练时自动禁用，退化为标准 float32，无需任何改动 |
| 预期收益 | 训练速度 **+50%~100%**，显存占用 **−25%~30%** |

---

## 各模块独立自测

```bash
# 测试数据加载（需先准备数据）
python dataset.py

# 测试网络结构（前向传播尺寸验证）
python model.py

# 测试损失函数（可微分性验证）
python loss.py
```

---

## 常见问题

**Q：Windows 下 DataLoader 卡死？**
> 设置 `--num_workers 0`

**Q：显存不足？**
> AMP 混合精度训练已默认集成，GPU 上会自动启用 float16 前向传播，可节省约 25%~30% 显存。
> 若仍不足，可进一步减小 `--batch_size`（4 或 2）。

**Q：SSIM 损失为负数？**
> 检查归一化方式，确保 `data_range=2.0`（对应 `[-1,1]`）或 `1.0`（对应 `[0,1]`）与实际一致
