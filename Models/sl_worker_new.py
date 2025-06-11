import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as F
import torch.nn.functional as F_nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from functools import partial
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, f1_score


# 数据加载
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.npy_paths = []  # 存储.npy文件路径
        self.labels = []  # 存储对应标签

        for label in sorted(os.listdir(root_dir)):
            label_path = os.path.join(root_dir, label)
            if os.path.isdir(label_path) and label.isdigit():
                # 获取当前标签下.npy文件
                npy_files = glob.glob(os.path.join(label_path, "*.npy"))
                self.npy_paths.extend(npy_files)
                self.labels.extend([int(label)] * len(npy_files))

        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.npy_paths)

    def __getitem__(self, idx):
        npy_path = self.npy_paths[idx]
        depth_images = np.load(npy_path).astype(np.float32)
        depth_images /= 255

        if depth_images.shape[0] != 3:
            raise ValueError(f"Invalid npy file {npy_path}: expected 3 time steps, got {depth_images.shape[0]}")

        # 转换为 (3, H, W) 的 tensor
        image_tensor = torch.from_numpy(depth_images).unsqueeze(1)

        if self.transform:
            image_tensor = self.transform(image_tensor)

        label = self.labels[idx]  # 获取当前样本的标签
        return image_tensor, label

#高斯噪声
class AddGaussianNoise:
    def __init__(self, mean=0.0, std_range=(0.003, 0.01)):
        """
        :param std_range: 噪声标准差范围（随机取值），避免固定强度导致过拟合
        """
        self.mean = mean
        self.std_range = std_range

    def __call__(self, tensor):
        # 生成单帧噪声 [1, H, W]
        base_noise = torch.randn(1, *tensor.shape[-2:])

        # 沿时间维度复制 [T, H, W]
        noise = base_noise.repeat(tensor.shape[0], 1, 1)

        # 统一强度控制
        std = torch.rand(1) * (self.std_range[1] - self.std_range[0]) + self.std_range[0]
        return (tensor + noise * std).clamp(0, 1)

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std_range={self.std_range})"

# 光照变化
class RandomBrightnessContrast:
    def __init__(self, brightness_range=(0.7, 1.3), contrast_range=(0.7, 1.3)):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range

    def __call__(self, tensor):
        brightness_factor = torch.FloatTensor(1).uniform_(*self.brightness_range)
        contrast_factor = torch.FloatTensor(1).uniform_(*self.contrast_range)
        tensor = F.adjust_brightness(tensor, brightness_factor)
        tensor = F.adjust_contrast(tensor, contrast_factor)
        return tensor.clamp(0, 1)

# 数据增强变换
class TemporalTransform:
    def __init__(self, transform):
        self.transform = transform  # 单帧变换

    def __call__(self, x):
        # 合并时间维度和通道维度（T*C, H, W）
        t, c, h, w = x.shape
        merged = x.reshape(t * c, h, w)  # 形状变为 (3, H, W)（假设C=1）

        # 应用变换
        transformed_merged = self.transform(merged)

        # 恢复时间维度和通道维度
        transformed = transformed_merged.reshape(t, c, *transformed_merged.shape[1:])
        return transformed


# 通道注意力
class SEAttention(nn.Module):

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(self.avg_pool(x))

# 空间注意力机制
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        concat = torch.cat([avg_pool, max_pool], dim=1)
        return x * self.sigmoid(self.conv(concat))


# 残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1,kernel_size=3,
                 use_se=True, use_spatial=False, reduction=16,dropout_p=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

        self.dropout = nn.Dropout(dropout_p) if in_channels>=64 else None

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=(kernel_size-1)//2, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        attn_layers = []
        if use_se:
            attn_layers.append(SEAttention(out_channels, reduction))
        if use_spatial:
            attn_layers.append(SpatialAttention())
        self.attention = nn.Sequential(*attn_layers) if attn_layers else nn.Identity()

        self.residual = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = self.residual(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.attention(x)
        x += residual
        return self.act(x)


# 共享基础卷积层
class BaseEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.MaxPool2d(2),  # 输入224→56
            SpatialAttention(),

            nn.Conv2d(24, 48, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.MaxPool2d(2),  # 56→28
            SEAttention(48),
            SpatialAttention(),

            nn.Conv2d(48, 96, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.MaxPool2d(2),  # 28→14
            SEAttention(96),

        )

    def forward(self, x):
        return self.layers(x)


# 时序空间模型
class TemporalSpatialModel(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        # 共享基础卷积层
        self.base_encoder = BaseEncoder()

        # 空间增强分支
        self.spatial_enhancer = nn.Sequential(

            nn.Conv2d(96,192, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(192),
            nn.GELU(),
            SEAttention(192),
            nn.Dropout(0.2),

            nn.AdaptiveAvgPool2d(1)
        )

        # 时间分支
        self.temporal_post_encoder = nn.Sequential(

            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Flatten(1)
        )

        # 时间卷积
        self.temporal_conv = nn.Sequential(

            nn.Conv1d(96*7*7, 192, kernel_size=2),
            nn.BatchNorm1d(192),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Conv1d(192, 96, kernel_size=2),
            nn.BatchNorm1d(96),
            nn.GELU(),

        )


        # 特征融合
        self.fusion_block = nn.Sequential(
            nn.Linear(192+96, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3)
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # 输入形状[B,3,1,224,224]
        img_t2, img_t1, img_t = x[:,0], x[:,1], x[:,2]

        # 共享基础卷积层
        feat_t2 = self.base_encoder(img_t2)
        feat_t1 = self.base_encoder(img_t1)
        feat_t = self.base_encoder(img_t)

        # 空间主分支增强
        spatial_feat = self.spatial_enhancer(feat_t)  #
        spatial_feat = spatial_feat.flatten(1)

        # 时间辅助分支
        t2_feat = self.temporal_post_encoder(feat_t2).flatten(1)
        t1_feat = self.temporal_post_encoder(feat_t1).flatten(1)
        t_feat = self.temporal_post_encoder(feat_t).flatten(1)

        # 拼接时间维度
        temporal_seq = torch.stack([t2_feat, t1_feat, t_feat], dim=2)

        temporal_feat = self.temporal_conv(temporal_seq)

        # 特征融合
        fused_feat = torch.cat([
            spatial_feat,
            temporal_feat.flatten(1)
        ], dim=1)
        fused_feat = self.fusion_block(fused_feat).flatten(1)

        return self.classifier(fused_feat)

    def extract_features(self, x):
        # 输入形状[B,3,1,224,224]
        img_t2, img_t1, img_t = x[:, 0], x[:, 1], x[:, 2]

        # 共享基础卷积层
        feat_t2 = self.base_encoder(img_t2)
        feat_t1 = self.base_encoder(img_t1)
        feat_t = self.base_encoder(img_t)

        # 空间主分支增强
        spatial_feat = self.spatial_enhancer(feat_t)  #
        spatial_feat = spatial_feat.flatten(1)

        # 时间辅助分支
        t2_feat = self.temporal_post_encoder(feat_t2).flatten(1)
        t1_feat = self.temporal_post_encoder(feat_t1).flatten(1)
        t_feat = self.temporal_post_encoder(feat_t).flatten(1)

        # 拼接时间维度
        temporal_seq = torch.stack([t2_feat, t1_feat, t_feat], dim=2)
        temporal_feat = self.temporal_conv(temporal_seq)

        # 特征融合
        fused_feat = torch.cat([
            spatial_feat,
            temporal_feat.flatten(1)
        ], dim=1)
        fused_feat = self.fusion_block(fused_feat).flatten(1)

        return fused_feat

# 自定义支持变换的子集类
class TransformedSubset(torch.utils.data.Subset):
    def __init__(self, dataset, indices, transform=None):
        super().__init__(dataset, indices)
        self.transform = transform

    def __getitem__(self, idx):
        x, y = super().__getitem__(idx)
        if self.transform is not None:
            x = self.transform(x)
        return x, y

# 训练
def train():
    data_root = "traindata"
    batch_size = 32
    num_epochs = 100
    lr = 0.0005
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据增强
    transform = transforms.Compose([
        # 下采样
        TemporalTransform(transforms.Resize((224, 224))),
        # 随机旋转
        TemporalTransform(transforms.RandomRotation(degrees=10)),
        # 随机仿射变换
        TemporalTransform(transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1))),
        # 随机高斯噪声
        TemporalTransform(transforms.RandomApply([AddGaussianNoise(std_range=(0.005, 0.015))], p=0.3)),
        # 随机光照变化
        TemporalTransform(transforms.RandomApply([RandomBrightnessContrast()], p=0.3))
    ])

    val_transform = transforms.Compose([
        # 下采样
        TemporalTransform(transforms.Resize((224, 224)))
    ])

    # 加载三帧数据集
    train_dataset = ImageDataset(root_dir=data_root, transform=transform)  # 训练集使用增强变换
    val_dataset = ImageDataset(root_dir=data_root, transform=val_transform)  # 验证集使用验证变换

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_indices, val_indices = torch.utils.data.random_split(
        list(range(len(train_dataset))),
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    train_dataset = TransformedSubset(train_dataset, train_indices.indices)
    val_dataset = TransformedSubset(val_dataset, val_indices.indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 优化器参数分组（权重衰减）
    def get_optim_params(model):
        decay_params, no_decay_params = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'bias' in name or 'bn' in name or 'norm' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        return [
            {'params': decay_params, 'weight_decay': 5e-5},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]

    # 初始化模型
    model = TemporalSpatialModel(num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(get_optim_params(model), lr=lr, betas=(0.9, 0.999))

    # 学习率调度（warmup+余弦退火重启）
    warmup_epochs = 10
    eta_min = 1e-6  # 最小学习率
    T_0 = 30  # 第一次重启的周期
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs),
            CosineAnnealingWarmRestarts(optimizer, T_0=T_0, eta_min=eta_min)
        ],
        milestones=[warmup_epochs]
    )

    # 训练监控
    epoch_losses, epoch_val_losses = [], []
    epoch_accs, epoch_val_accs = [], []
    epoch_val_macro_f1 = []
    best_macro_f1 = -float('inf')
    patience = 20
    no_improve_epochs = 0
    best_model_weights = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if (batch_idx + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}, Acc: {100 * correct / total:.2f}%')

        # 训练指标
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        epoch_losses.append(epoch_loss)
        epoch_accs.append(epoch_acc)

        # 验证阶段（收集所有预测和标签）
        model.eval()
        val_loss_total = 0.0
        all_val_labels = []
        all_val_preds = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss_total += loss.item()

                # 收集标签和预测结果（转换为CPU的numpy数组）
                _, predicted = torch.max(outputs.data, 1)
                all_val_labels.extend(labels.cpu().numpy())
                all_val_preds.extend(predicted.cpu().numpy())

        # 计算验证指标
        epoch_val_loss = val_loss_total / len(val_loader)
        val_acc = 100 * sum(np.array(all_val_preds) == np.array(all_val_labels)) / len(all_val_labels)
        macro_f1 = f1_score(all_val_labels, all_val_preds, average='macro')  # 宏F1

        # 记录指标
        epoch_val_losses.append(epoch_val_loss)
        epoch_val_accs.append(val_acc)
        epoch_val_macro_f1.append(macro_f1)

        # 学习率调度
        scheduler.step()

        # 打印分类报告（每类指标）
        print(f"\n{'验证集分类报告':^40}")
        print(classification_report(all_val_labels, all_val_preds,
                                    target_names=[f"类别{i}" for i in range(5)],
                                    digits=4))
        # 打印汇总指标
        print(f'Epoch {epoch + 1} 完成: Train Loss {epoch_loss:.4f}, Acc {epoch_acc:.2f}% | '
              f'Val Loss {epoch_val_loss:.4f}, Acc {val_acc:.2f}%, Macro F1 {macro_f1:.4f}')

        # 基于宏F1的早停机制
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_model_weights = model.state_dict().copy()
            no_improve_epochs = 0
            print(f'验证宏F1提升，保存最佳模型（当前最佳：{best_macro_f1:.4f}）')
        else:
            no_improve_epochs += 1
            print(f'验证宏F1未提升，连续{no_improve_epochs}轮无改进（当前最佳：{best_macro_f1:.4f}）')

        if no_improve_epochs >= patience:
            print(f'\n早停触发！当前最佳宏F1：{best_macro_f1:.4f}')
            break

    # 保存最佳模型
    if best_model_weights:
        torch.save(best_model_weights,
                   "weights/newmodel.pth")
        print(f'最佳模型已保存（验证宏F1：{best_macro_f1:.4f}）')

    print('训练完成！')


if __name__ == '__main__':
    train()