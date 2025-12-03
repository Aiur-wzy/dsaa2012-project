#!/usr/bin/env python
# coding: utf-8
# Auto-generated from notebook: draft.ipynb

# %% [markdown] cell 0
# A Minimal Real-Time Facial Expression Recognition System: Lightweight CNN on FER-2013 with OpenCV-based Inference

# %% [markdown] cell 1
# 1.数据集准备与校验

# %% cell 2
import os
import pandas as pd
from collections import Counter

FER_CSV_PATH = "fer2013.csv"  # 修改为你的实际路径

EMOTION_LABELS = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

df = pd.read_csv(FER_CSV_PATH)

print(df.head())
print("Total samples:", len(df))
print("Usage split:", df["Usage"].value_counts())

def describe_split(name):
    sub = df[df["Usage"] == name]
    cnt = Counter(sub["emotion"].values.tolist())
    print(f"\n=== {name} ===")
    total = len(sub)
    for k in sorted(cnt.keys()):
        print(f"{k} ({EMOTION_LABELS[k]}): {cnt[k]} ({cnt[k]/total:.3%})")
    print("Total:", total)

for usage in ["Training", "PublicTest", "PrivateTest"]:
    if usage in df["Usage"].unique():
        describe_split(usage)

# %% [markdown] cell 3
# 2.数据加载与预处理 
# 自定义 Dataset（48×48 灰度读取、归一化、通道适配）
# FER-2013 的 pixels 是空格分隔的 48*48=2304 个像素值（0–255）
# 我们将其读为 numpy 数组，reshape 为 48×48
# 归一化到 [0,1] 再做标准化（使用 ImageNet 或自定义均值方差）
# 通道：
# 若用自定义 CNN：可直接使用单通道 (1×48×48)
# 若想用预训练模型（ResNet 等）：将灰度复制到 3 通道 (3×48×48)
# 这里提供两种通道模式开关：in_chans = 1 或 3。

# %% cell 4
import numpy as np
import torch
from torch.utils.data import Dataset

class FER2013Dataset(Dataset):
    def __init__(self, df, usage_filter=None, transform=None, in_chans=1):
        """
        df: pandas DataFrame, 读取自 fer2013.csv
        usage_filter: "Training" / "PublicTest" / "PrivateTest" / None
        transform: Albumentations 或 torchvision 风格的变换
        in_chans: 1 (灰度) 或 3 (复制到 RGB 形式)
        """
        if usage_filter is not None:
            df = df[df["Usage"] == usage_filter].reset_index(drop=True)
        self.df = df
        self.transform = transform
        self.in_chans = in_chans

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = int(row["emotion"])
        pixels = np.fromstring(row["pixels"], dtype=np.uint8, sep=" ")
        img = pixels.reshape(48, 48)  # H, W  (灰度)

        # Albumentations 需要 HWC
        img = np.expand_dims(img, axis=-1)  # (48, 48, 1)

        if self.in_chans == 3:
            img = np.repeat(img, 3, axis=-1)  # (48, 48, 3)

        if self.transform is not None:
            # Albumentations 风格
            transformed = self.transform(image=img)
            img = transformed["image"]
        else:
            # 转为 torch
            img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

        return img, label

# %% [markdown] cell 5
# 3.数据增强设计（鲁棒性与噪声）
# 这里使用 Albumentations，特点：
# 
# 几何增强：随机平移、旋转、小尺度仿射、水平翻转
# 光照变化：亮度/对比度抖动、Gamma
# 模糊/压缩：高斯模糊、JPEG 压缩
# 随机遮挡（Cutout 类似）
# MixUp / CutMix：通常在 batch 级别完成，在训练循环中实现

# %% cell 6
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transform(in_chans=1):
    # 归一化参数：灰度时可使用 mean=0.5, std=0.5；3通道可统一 0.5/0.5
    if in_chans == 1:
        mean = (0.5,)
        std = (0.5,)
    else:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

    return A.Compose([
        # 几何增强
        A.ShiftScaleRotate(
            shift_limit=0.1, scale_limit=0.1, rotate_limit=15,
            border_mode=0, value=0, p=0.7
        ),
        A.HorizontalFlip(p=0.5),  # 表情相对对称，可以用

        # 光照增强
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),

        # 模糊/压缩
        A.OneOf([
            A.GaussianBlur(blur_limit=3),
            A.MotionBlur(blur_limit=3),
        ], p=0.3),
        A.JpegCompression(quality_lower=60, quality_upper=100, p=0.3),

        # 随机遮挡（类似 Cutout，模拟眼镜/口罩）
        A.CoarseDropout(
            max_holes=2,
            max_height=12,
            max_width=12,
            min_holes=1,
            fill_value=0,
            p=0.5
        ),

        # 最后归一化+转 Tensor
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

def get_val_transform(in_chans=1):
    if in_chans == 1:
        mean = (0.5,)
        std = (0.5,)
    else:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

    return A.Compose([
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

# %% [markdown] cell 7
# MixUp / CutMix（缓解标签噪声）

# %% cell 8
import random

def mixup_data(x, y, alpha=0.2):
    """返回混合后的数据与对应标签系数"""
    if alpha <= 0:
        return x, y, torch.ones_like(y, dtype=torch.float32), torch.arange(len(y))
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# %% [markdown] cell 9
# 轻量 CNN 模型设计

# %% cell 10
import torch.nn as nn
import torch.nn.functional as F
import torch

class DWConvBlock(nn.Module):
    """Depthwise Separable Conv 块：DWConv + PWConv"""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride,
                            padding=1, groups=in_ch, bias=False)
        self.dw_bn = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.dw(x)
        x = self.dw_bn(x)
        x = self.relu(x)
        x = self.pw(x)
        x = self.pw_bn(x)
        x = self.relu(x)
        return x

class LightweightFERNet(nn.Module):
    def __init__(self, in_chans=1, num_classes=7, width_mult=1.0):
        super().__init__()
        def c(ch):  # 调整通道数
            return int(ch * width_mult)

        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, c(32), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c(32)),
            nn.ReLU(inplace=True),
        )

        # 48x48 -> 24x24
        self.block1 = nn.Sequential(
            DWConvBlock(c(32), c(64), stride=2),
            DWConvBlock(c(64), c(64), stride=1),
        )
        # 24x24 -> 12x12
        self.block2 = nn.Sequential(
            DWConvBlock(c(64), c(128), stride=2),
            DWConvBlock(c(128), c(128), stride=1),
        )
        # 12x12 -> 6x6
        self.block3 = nn.Sequential(
            DWConvBlock(c(128), c(256), stride=2),
            DWConvBlock(c(256), c(256), stride=1),
        )
        # 6x6 -> 3x3
        self.block4 = nn.Sequential(
            DWConvBlock(c(256), c(512), stride=2),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)  # 全局平均池化 -> (B, C, 1, 1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(c(512), num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x)      # (B, C, 1, 1)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# 参数量估算
if __name__ == "__main__":
    model = LightweightFERNet(in_chans=1, num_classes=7, width_mult=0.75)
    x = torch.randn(1, 1, 48, 48)
    y = model(x)
    print("Output shape:", y.shape)
    print("Total params:", sum(p.numel() for p in model.parameters())/1e6, "M")

# %% [markdown] cell 11
# 6.损失函数、类别不均衡与优化设置

# %% [markdown] cell 12
# 6.1Label Smoothing 交叉熵

# %% cell 13
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 可根据类别不均衡指定 class weights
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

classes = np.arange(7)
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=df[df["Usage"]=="Training"]["emotion"].values
)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

criterion_ce = nn.CrossEntropyLoss(
    weight=class_weights_tensor,  # 若不想加权，可以设为 None
    label_smoothing=0.6
)

# %% [markdown] cell 14
# 6.2 Focal Loss（可选）

# %% cell 15
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        if isinstance(alpha, (list, np.ndarray)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, reduction="none",
                             weight=self.alpha.to(logits.device) if self.alpha is not None else None)
        pt = torch.exp(-ce)
        focal_loss = (1 - pt) ** self.gamma * ce
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss

# 使用范例：
# criterion = FocalLoss(gamma=2.0, alpha=class_weights_tensor)
criterion = criterion_ce  # 先用带 label smoothing + class weight 的 CE

# %% [markdown] cell 16
# 优化器与学习率调度

# %% cell 17
model = LightweightFERNet(in_chans=1, num_classes=7, width_mult=0.75).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)  # T_max=epochs

# %% [markdown] cell 18
# 7.Dataloader 与训练细节（AMP / 重采样 / 指标）

# %% [markdown] cell 19
# 7.1 DataLoader 与重采样策略

# %% cell 20
from torch.utils.data import DataLoader, WeightedRandomSampler

in_chans = 1  # 或 3
train_transform = get_train_transform(in_chans=in_chans)
val_transform = get_val_transform(in_chans=in_chans)

train_dataset = FER2013Dataset(df, usage_filter="Training",
                               transform=train_transform, in_chans=in_chans)
val_dataset = FER2013Dataset(df, usage_filter="PublicTest",
                             transform=val_transform, in_chans=in_chans)
test_dataset = FER2013Dataset(df, usage_filter="PrivateTest",
                              transform=val_transform, in_chans=in_chans)

# 可选的重采样
train_labels = train_dataset.df["emotion"].values
class_sample_count = np.array([len(np.where(train_labels == t)[0]) for t in range(7)])
weights = 1.0 / class_sample_count
samples_weight = np.array([weights[t] for t in train_labels])
samples_weight = torch.from_numpy(samples_weight).double()
sampler = WeightedRandomSampler(weights=samples_weight,
                                num_samples=len(samples_weight),
                                replacement=True)

batch_size = 128
use_sampler = False  # 若想启用重采样，改为 True

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    sampler=sampler if use_sampler else None,
    shuffle=not use_sampler,
    num_workers=4,
    pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False,
    num_workers=4, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False,
    num_workers=4, pin_memory=True
)

# %% [markdown] cell 21
# 训练循环

# %% cell 22
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm

scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
epochs = 40
mixup_alpha = 0.2  # 若不想 MixUp，可设为 0

def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    pbar = tqdm(loader, desc=f"Train Epoch {epoch}")
    for imgs, labels in pbar:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        if mixup_alpha > 0:
            inputs, targets_a, targets_b, lam = mixup_data(imgs, labels, alpha=mixup_alpha)
        else:
            inputs, targets_a, targets_b, lam = imgs, labels, labels, 1.0

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            outputs = model(inputs)
            if mixup_alpha > 0:
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * imgs.size(0)

        # 统计预测（注意：使用原标签 labels）
        preds = outputs.argmax(dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro")

    return avg_loss, acc, macro_f1

@torch.no_grad()
def evaluate(model, loader, criterion, device, desc="Val"):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for imgs, labels in tqdm(loader, desc=desc):
        imgs = imgs.to(device)
        labels = labels.to(device)

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro")
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(7)))

    return avg_loss, acc, macro_f1, cm

# %% [markdown] cell 23
# 训练主循环

# %% cell 24
best_val_acc = 0.0
patience = 7
no_improve_epochs = 0
best_model_path = "best_fer_model.pth"

for epoch in range(1, epochs + 1):
    train_loss, train_acc, train_f1 = train_one_epoch(
        model, train_loader, optimizer, criterion_ce, device, epoch
    )
    val_loss, val_acc, val_f1, val_cm = evaluate(
        model, val_loader, criterion_ce, device, desc="Val"
    )

    scheduler.step()

    print(f"\nEpoch {epoch}/{epochs}")
    print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1(macro): {train_f1:.4f}")
    print(f"Val   Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1(macro): {val_f1:.4f}")
    print("Val Confusion Matrix:\n", val_cm)

    # 早停策略 + 保存最好模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        no_improve_epochs = 0
        torch.save(model.state_dict(), best_model_path)
        print(">> Saved new best model.")
    else:
        no_improve_epochs += 1
        if no_improve_epochs >= patience:
            print(">> Early stopping triggered.")
            break

print("Best Val Acc:", best_val_acc)

# %% [markdown] cell 25
# 最终测试集评估

# %% cell 26
# 加载最佳模型
model.load_state_dict(torch.load(best_model_path, map_location=device))

test_loss, test_acc, test_f1, test_cm = evaluate(
    model, test_loader, criterion_ce, device, desc="Test"
)
print("\n=== Test Results ===")
print(f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, F1(macro): {test_f1:.4f}")
print("Test Confusion Matrix:\n", test_cm)

# 逐类 F1
y_true_all = []
y_pred_all = []
model.eval()
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            outputs = model(imgs)
        preds = outputs.argmax(dim=1).cpu().numpy()
        y_pred_all.extend(preds)
        y_true_all.extend(labels.cpu().numpy())

per_class_f1 = f1_score(y_true_all, y_pred_all, average=None, labels=list(range(7)))
for i, f1v in enumerate(per_class_f1):
    print(f"Class {i} ({EMOTION_LABELS[i]}): F1 = {f1v:.4f}")

# %% [markdown] cell 27
# 模型压缩与部署优化

# %% [markdown] cell 28
# 知识蒸馏（ResNet-18 → 轻量 CNN 学生）
# 思路：
# 
# 教师：预训练或在 FER 上微调的 ResNet-18（3 通道）
# 学生：你已有的 LightweightFERNet（1 或 3 通道）
# 总损失 = 真实标签交叉熵 + 蒸馏 KL 散度
# 超参：温度 T、蒸馏权重 alpha

# %% cell 29
import torchvision.models as models
import torch.nn as nn
import torch

num_classes = 7

teacher = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
teacher.fc = nn.Linear(teacher.fc.in_features, num_classes)
teacher.load_state_dict(torch.load("resnet18_fer.pth"))  # 你预先训练好的权重
teacher.eval().to(device)
for p in teacher.parameters():
    p.requires_grad = False

# %% [markdown] cell 30
# 蒸馏损失

# %% cell 31
import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits, targets, T=4.0, alpha=0.5, ce_weight=1.0):
    """
    student_logits, teacher_logits: (B, num_classes)
    targets: ground truth labels
    """
    # 硬标签交叉熵
    ce = F.cross_entropy(student_logits, targets) * ce_weight

    # 软标签 KL 散度
    # 注意：KLDivLoss 默认 input 是 log-prob，target 是 prob
    log_p_s = F.log_softmax(student_logits / T, dim=1)
    p_t = F.softmax(teacher_logits / T, dim=1)
    kd = F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)

    return alpha * kd + (1 - alpha) * ce

# %% [markdown] cell 32
# 蒸馏训练循环

# %% cell 33
student = LightweightFERNet(in_chans=3, num_classes=num_classes, width_mult=0.75).to(device)
optimizer = torch.optim.AdamW(student.parameters(), lr=5e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)

T = 4.0
alpha = 0.7

def train_one_epoch_kd(student, teacher, loader, optimizer, device, epoch):
    student.train()
    teacher.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    pbar = tqdm(loader, desc=f"KD Train Epoch {epoch}")
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.no_grad():
            teacher_logits = teacher(imgs)

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            student_logits = student(imgs)
            loss = distillation_loss(student_logits, teacher_logits, labels, T=T, alpha=alpha)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * imgs.size(0)
        preds = student_logits.argmax(dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro")
    return avg_loss, acc, macro_f1

# %% [markdown] cell 34
#  剪枝与稀疏化（结构化通道剪枝）
# 思路：
# 
# 在训练/蒸馏阶段，对卷积层添加 L1 正则促使通道稀疏
# 使用类似 torch.nn.utils.prune.ln_structured 对卷积层按通道剪枝
# 剪枝后进行短期微调恢复精度

# %% [markdown] cell 35
#  引入 L1 稀疏正则

# %% cell 36
def l1_channel_sparsity(model, lambda_l1=1e-5):
    l1_loss = 0.0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            l1_loss += m.weight.abs().sum()
    return lambda_l1 * l1_loss

# %% [markdown] cell 37
# 训练循环

# %% cell 38
with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
    student_logits = student(imgs)
    loss_main = distillation_loss(student_logits, teacher_logits, labels, T=T, alpha=alpha)
    loss = loss_main + l1_channel_sparsity(student, lambda_l1=1e-5)

# %% [markdown] cell 39
# 通道剪枝
# 建议：先用较小剪枝率（0.2–0.3）试水，剪后再用小学习率（如 1e-4）微调 10–20 个 epoch

# %% cell 40
import torch.nn.utils.prune as prune

def structured_prune_model(model, amount=0.3):
    """
    对所有 Conv2d 按输出通道 L1-norm 剪枝 amount 比例
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(
                module,
                name="weight",
                amount=amount,
                n=1,         # L1
                dim=0        # 沿输出通道剪
            )
            prune.remove(module, "weight")  # 使剪枝永久化
    return model

# 使用示例：
student = structured_prune_model(student, amount=0.3)  # 剪 30% 输出通道

# %% [markdown] cell 41
#  量化（PTQ / QAT）

# %% [markdown] cell 42
# 3.1 推理后量化（Post-Training Quantization, PTQ, INT8）
# PyTorch 动态量化（对 Linear, LSTM 等）很方便，CNN 的真正 INT8 部署通常要依赖后端（如 ONNX Runtime / TensorRT）。给一个基础的动态量化示例（对最后 FC 层）：

# %% cell 43
quantized_model = torch.quantization.quantize_dynamic(
    student.cpu(),
    {nn.Linear},  # 只对 Linear 做动态量化
    dtype=torch.qint8
)
torch.save(quantized_model.state_dict(), "student_quant_dynamic.pth")

# %% [markdown] cell 44
# 3.2 量化感知训练（QAT，简略示例）
# PyTorch QAT 流程简要：

# %% cell 45
from torch.ao.quantization import (
    get_default_qat_qconfig,
    prepare_qat,
    convert
)

student_qat = LightweightFERNet(in_chans=1, num_classes=7, width_mult=0.75)
student_qat.train()

student_qat.qconfig = get_default_qat_qconfig("fbgemm")
student_qat = prepare_qat(student_qat)   # 插入 FakeQuantize 模块

# 用你原来的训练循环再训练 5–10 个 epoch（lr 可小一些）
# ...

student_qat.eval()
student_int8 = convert(student_qat)
torch.save(student_int8.state_dict(), "student_qat_int8.pth")

# %% [markdown] cell 46
# 4. 导出 ONNX / OpenVINO / TFLite
# 4.1 导出 ONNX

# %% cell 47
dummy = torch.randn(1, 1, 48, 48, device=device)
student.eval().to(device)
torch.onnx.export(
    student,
    dummy,
    "fer_student.onnx",
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=13
)

# %% cell 48
#在 PC 上可用 ONNX Runtime
import onnxruntime as ort
import numpy as np

ort_session = ort.InferenceSession("fer_student.onnx", providers=["CPUExecutionProvider"])

def onnx_infer(batch_np):  # batch_np: (B,1,48,48) float32
    ort_inputs = {"input": batch_np}
    logits = ort_session.run(["logits"], ort_inputs)[0]
    return logits

# %% [markdown] cell 49
# OpenVINO（x86/Intel）
# mo --input_model fer_student.onnx --input_shape [1,1,48,48] --data_type FP16 --output_dir openvino_model

# %% [markdown] cell 50
# 4.3 TFLite（移动端）
# 在 Python 用 torch.onnx 导出后，用 onnx-tf 转成 TF，然后使用 TFLite Converter；或
# 直接重构一个 TF/Keras 版本的网络，在 Keras 里训练/加载 PyTorch 权重，然后：

# %% cell 51
4.3 TFLite（移动端）
在 Python 用 torch.onnx 导出后，用 onnx-tf 转成 TF，然后使用 TFLite Converter；或
直接重构一个 TF/Keras 版本的网络，在 Keras 里训练/加载 PyTorch 权重，然后：

# %% cell 52
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model("saved_model_dir")
# PTQ 示例：float16 或 INT8
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
open("fer_student.tflite", "wb").write(tflite_model)

# %% [markdown] cell 53
# 4.4 精度-延迟-体积记录
# 对于每种模型（Teacher, Student, Pruned, INT8）记录：
# 
# 模型大小：os.path.getsize("model.xxx") / 1024**2
# 测试集准确率 / macro F1
# 平均推理时间（多次前向取均值）

# %% cell 54
import time, os

def benchmark_model(model, dataloader, device, n_warmup=10, n_runs=100):
    model.eval().to(device)
    # 预热
    with torch.no_grad():
        for i, (x, _) in enumerate(dataloader):
            x = x.to(device)
            _ = model(x)
            if i >= n_warmup:
                break
    # 正式计时
    times = []
    with torch.no_grad():
        for i, (x, _) in enumerate(dataloader):
            x = x.to(device)
            t0 = time.time()
            _ = model(x)
            t1 = time.time()
            times.append((t1 - t0) / x.size(0))
            if i >= n_runs:
                break
    return sum(times) / len(times)

model_size_mb = os.path.getsize("student.pth") / (1024**2)
latency = benchmark_model(student, test_loader, device)
print(f"Size: {model_size_mb:.2f} MB,  Avg Latency: {latency*1000:.2f} ms / image")

# %% [markdown] cell 55
# 人脸检测与对齐（OpenCV 链路）
# 1. 人脸检测器选择与对比
# 两种常用方案：
# 
# Haar/LBP 级联（经典，CPU 轻量，但对小脸和侧脸较弱）
# DNN Res10 SSD (deploy.prototxt + res10_300x300_ssd_iter_140000.caffemodel)，OpenCV cv2.dnn，精度和鲁棒性更好，但略慢。

# %% [markdown] cell 56
# 1.1 Haar 检测器

# %% cell 57
import cv2

haar_face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_faces_haar(gray_frame):
    faces = haar_face.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    # 返回 [x,y,w,h] 列表
    return faces

# %% [markdown] cell 58
# 1.2 DNN Res10 SSD

# %% cell 59
dnn_proto = "deploy.prototxt"
dnn_model = "res10_300x300_ssd_iter_140000.caffemodel"

net = cv2.dnn.readNetFromCaffe(dnn_proto, dnn_model)

def detect_faces_dnn(frame, conf_threshold=0.5):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (x1, y1, x2, y2) = box.astype("int")
            boxes.append((x1, y1, x2-x1, y2-y1))
    return boxes

# %% [markdown] cell 60
# 2. 人脸对齐
# 常见做法：
# 
# 使用 5 点或 68 点关键点（如 dlib / retinaface / opencv face landmarks）
# 将双眼中心连线旋转到水平，平移/缩放对齐到标准模板
# 这里给一个简化版本（假设你已有 68 点或 5 点关键点 landmarks）：

# %% cell 61
import numpy as np

def align_face(img, landmarks, output_size=(48, 48)):
    # landmarks: shape (5,2) 或 (68,2)，这里假设 5 点：左眼、右眼、鼻尖、嘴左、嘴右
    desired_left_eye = (0.35, 0.35)
    desired_face_width, desired_face_height = output_size

    left_eye, right_eye = landmarks[0], landmarks[1]
    # 计算眼睛中心与角度
    eye_center = ((left_eye[0]+right_eye[0]) / 2.0,
                  (left_eye[1]+right_eye[1]) / 2.0)
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))  # 逆时针角度

    # 期望眼距
    dist = np.sqrt((dx ** 2) + (dy ** 2))
    desired_dist = (1.0 - 2*desired_left_eye[0]) * desired_face_width
    scale = desired_dist / dist

    # 获取仿射变换矩阵
    eyes_center = eye_center
    M = cv2.getRotationMatrix2D(eyes_center, angle, scale)

    # 调整平移
    tX = desired_face_width * 0.5
    tY = desired_face_height * desired_left_eye[1]
    M[0, 2] += (tX - eyes_center[0])
    M[1, 2] += (tY - eyes_center[1])

    aligned = cv2.warpAffine(img, M, (desired_face_width, desired_face_height),
                             flags=cv2.INTER_CUBIC)
    return aligned

# %% [markdown] cell 62
# 如果你暂时没有关键点模型，可以先直接依赖检测框：

# %% cell 63
def crop_face(frame, box, margin=0.2, out_size=48):
    x, y, w, h = box
    cx, cy = x + w//2, y + h//2
    side = int(max(w, h) * (1 + margin))
    x1 = max(cx - side//2, 0)
    y1 = max(cy - side//2, 0)
    x2 = min(cx + side//2, frame.shape[1])
    y2 = min(cy + side//2, frame.shape[0])
    face = frame[y1:y2, x1:x2]
    face = cv2.resize(face, (out_size, out_size))
    return face

# %% [markdown] cell 64
# 3. 批量与缓存优化、跟踪减少检测频率
# 对每 N 帧做一次完整检测，其余帧使用跟踪器（KCF/CSRT）跟踪人脸框
# 对每个检测到的脸维护一个 cv2.TrackerKCF_create() 或 TrackerCSRT_create()

# %% cell 65
def create_tracker():
    return cv2.legacy.TrackerKCF_create()  # 或 TrackerCSRT_create()

trackers = []  # 列表[(tracker, id), ...]

def update_trackers(frame):
    new_boxes = []
    for tracker, tid in trackers:
        ok, box = tracker.update(frame)
        if ok:
            new_boxes.append((tid, box))  # box: (x,y,w,h)
    return new_boxes

# %% [markdown] cell 66
# 实时推理原型（摄像头管线）
# 总体结构：
# 
# VideoCapture 读取摄像头
# 多线程：一个线程采集帧，一个线程做推理（可选）
# 使用上面的人脸检测 + 对齐
# 将对齐后的 48×48 灰度输入轻量 CNN / 量化模型
# 在画面上叠加边框、表情标签、置信度
# 键盘交互：切换检测器 / 对齐开关等

# %% cell 67
import cv2
import time
import numpy as np
import torch

# 假设你已经有: student (PyTorch 模型), val_transform 或推理预处理
EMOTION_LABELS = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]

use_dnn_detector = True  # 按键切换
use_alignment = False    # 是否启用对齐（若有关键点模块）
frame_skip_for_detect = 10  # 每10帧做一次检测
frame_count = 0

cap = cv2.VideoCapture(0)

# 简单预处理函数（与训练时一致）
def preprocess_face(face_bgr):
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(gray, (48, 48))
    # [0,1] & 标准化 (0.5, 0.5)
    face = face.astype(np.float32) / 255.0
    face = (face - 0.5) / 0.5
    face = face[None, None, :, :]  # (1,1,48,48)
    return torch.from_numpy(face).to(device)

student.eval().to(device)

prev_time = time.time()
fps_display = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    orig = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 1) 检测／跟踪
    if frame_count % frame_skip_for_detect == 1:
        if use_dnn_detector:
            face_boxes = detect_faces_dnn(frame)
        else:
            face_boxes = detect_faces_haar(gray)
        # TODO: 初始化 trackers 等，这里简化为每次直接用检测结果
    # 否则你可以用 tracker.update() 来更新 face_boxes

    # 2) 对每个检测到的人脸做推理
    if len(face_boxes) > 0:
        batch_faces = []
        crops = []
        for box in face_boxes:
            face_crop = crop_face(frame, box, margin=0.2, out_size=48)
            crops.append((box, face_crop))
            tensor = preprocess_face(face_crop)
            batch_faces.append(tensor)

        batch = torch.cat(batch_faces, dim=0)  # (N,1,48,48)

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = student(batch)
                probs = torch.softmax(logits, dim=1).cpu().numpy()

        # 3) 在图像上绘制结果
        for (box, face_crop), prob in zip(crops, probs):
            x, y, w, h = box
            label_id = prob.argmax()
            label = EMOTION_LABELS[label_id]
            conf = prob[label_id]

            cv2.rectangle(orig, (x, y), (x+w, y+h), (0,255,0), 2)
            text = f"{label}: {conf:.2f}"
            cv2.putText(orig, text, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # 4) 帧率统计
    now = time.time()
    dt = now - prev_time
    prev_time = now
    fps = 1.0 / dt
    fps_display = 0.9*fps_display + 0.1*fps  # 平滑
    cv2.putText(orig, f"FPS: {fps_display:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    cv2.imshow("FER Demo", orig)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC退出
        break
    elif key == ord('d'):  # 切换检测器
        use_dnn_detector = not use_dnn_detector
    elif key == ord('a'):  # 切换对齐
        use_alignment = not use_alignment

cap.release()
cv2.destroyAllWindows()

# %% [markdown] cell 68
# 鲁棒性与公平性评测
# 1. 受控鲁棒性测试
# 对验证/测试集施加可控扰动，然后在每种扰动条件下评测准确率、F1 与混淆矩阵

# %% cell 69
def add_brightness_contrast(img, alpha=1.0, beta=0):  # alpha:对比度, beta:亮度
    new = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return new

def add_gaussian_blur(img, ksize=3, sigma=1.0):
    return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma)

def jpeg_compress(img, quality=50):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, enc = cv2.imencode('.jpg', img, encode_param)
    dec = cv2.imdecode(enc, 1)
    return dec

# %% [markdown] cell 70
# 1.2 姿态扰动（yaw/pitch/roll）
# 在真实视频中通过头部转动收集样本较现实；若只基于 2D 图像，可通过仿射变换模拟一定 yaw/roll：

# %% cell 71
def random_rotate(img, max_angle=30):
    h, w = img.shape[:2]
    angle = np.random.uniform(-max_angle, max_angle)
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h))

# %% [markdown] cell 72
# 1.3 时间稳定性（预测抖动）
# 在实时视频上对同一人进行连续预测
# 记录每一帧的预测标签，计算随时间的变化次数（抖动）
# 可计算 top-2 准确率：若真实标签在 top2 置信度中记为正确

# %% cell 73
def topk_accuracy(logits, labels, k=2):
    topk = torch.topk(logits, k, dim=1).indices.cpu().numpy()
    labels = labels.cpu().numpy()
    correct = sum([1 if labels[i] in topk[i] else 0 for i in range(len(labels))])
    return correct / len(labels)

# %% [markdown] cell 74
# 2. 公平性与偏置评估（代理方式）
# 在缺乏真实人口统计标签的情况下，可以用一些代理特征：
# 
# 肤色估计：简单 RGB/HSV 分析或用预训练的人脸属性模型（如 FairFace）提取
# 性别呈现、年龄段：基于公开人脸属性模型的推理结果作为“proxy label”
# 将测试集分成若干子集（如浅色皮肤组/深色皮肤组、年轻/中年/老年）
# 在每个子集上分别计算：
# 
# TPR（真实阳性率）、FNR
# 准确率、macro F1
# 各组之间的差值 ΔTPR, ΔFNR, ΔAcc
# 示意代码（假设你给每张图打了 group_id）：

# %% cell 75
from collections import defaultdict

def group_metrics(y_true, y_pred, groups):
    metrics = {}
    for g in set(groups):
        idx = [i for i, gg in enumerate(groups) if gg == g]
        if not idx:
            continue
        yt = [y_true[i] for i in idx]
        yp = [y_pred[i] for i in idx]
        acc = accuracy_score(yt, yp)
        f1 = f1_score(yt, yp, average="macro")
        metrics[g] = {"acc": acc, "f1": f1}
    return metrics

# %% [markdown] cell 76
# 报告时需要明确说明：
# 
# 使用的是代理变量而非真实种族/性别；
# 评测结果只反映模型在这些代理下的差异，存在不确定性；
# 如有明显差异，应在改进数据多样性、重新训练或加权损失后再次评估。
# 3. 误用与安全边界声明（建议文案要点）
# 在项目文档或 UI 中，应明确写出：
# 
# 该模型只适用于娱乐、非安全关键场景；
# 模型对表情的识别不是对“情绪状态”的可靠判断，不能用于：
# 心理健康诊断
# 谈判、招聘、执法等重要决策
# 模型性能在以下情况下会明显下降：
# 极端光照、强遮挡（口罩、墨镜）、大幅姿态偏转
# 模糊或压缩严重的视频
# 声明潜在偏置风险：不同人群在训练数据中的覆盖度不同，可能导致性能差异；
# 推荐在生产环境中始终有人类监督，不应将其作为唯一决策依据。

