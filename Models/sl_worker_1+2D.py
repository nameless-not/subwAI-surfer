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


# 数据加载
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = r"C:\Users\xiang\OneDrive\桌面\subwayai\pythonProject\subwAI-surfer\traindata"
        self.transform = transform
        self.npy_paths = []  # 存储.npy文件路径
        self.labels = []  # 存储对应标签

        for label in sorted(os.listdir(root_dir)):
            label_path = os.path.join(root_dir, label)
            if os.path.isdir(label_path) and label.isdigit():
                # 获取当前标签下所有.npy文件
                npy_files = glob.glob(os.path.join(label_path, "*.npy"))
                self.npy_paths.extend(npy_files)
                self.labels.extend([int(label)] * len(npy_files))

        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.npy_paths)

    def __getitem__(self, idx):
        npy_path = self.npy_paths[idx]
        depth_images = np.load(npy_path).astype(np.float32)

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
        # 随机选择噪声强度（0.003~0.01，适配锐化后的深度图）
        std = torch.rand(1) * (self.std_range[1] - self.std_range[0]) + self.std_range[0]
        noise = torch.randn(tensor.size(), device=tensor.device) * std + self.mean
        # 限制噪声范围，避免覆盖关键特征（深度图通常0~1，故限制在±0.02）
        return (tensor + noise).clamp(0.0, 1.0)

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std_range={self.std_range})"

class TemporalTransform:
    def __init__(self, transform):
        self.transform = transform  # 单帧变换

    def __call__(self, x):
        # x 形状：(3, 1, H, W)
        transformed = []
        for frame in x:  # 遍历每个时间步
            # 确保输入形状为 (C, H, W)
            if frame.dim() == 4:
                frame = frame.squeeze(0)
            # 单帧变换（去除通道维度 → 应用变换 → 恢复通道维度）
            transformed_frame = self.transform(frame).unsqueeze(0)
            transformed.append(transformed_frame)
        return torch.cat(transformed, dim=0)  # 合并时间步


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
    def __init__(self, in_channels, out_channels, stride=1,
                 use_se=True, use_spatial=True, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
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
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.residual(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.attention(x)
        x += residual
        return self.act(x)


# 1+2D时序空间模型
class TemporalSpatialModel(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        # 空间主分支
        self.spatial_branch = nn.Sequential(
            ResidualBlock(1, 32, stride=2, use_se=True, use_spatial=True),  # [32,112,112]
            ResidualBlock(32, 64, stride=2, use_spatial=True),  # [64,56,56]
            ResidualBlock(64, 128, stride=2, use_se=True, use_spatial=True),  # [128,28,28]
            ResidualBlock(128, 256, stride=2,use_se=True),  # [256,14,14]
            ResidualBlock(256, 512, stride=2, use_se=True),  # [512,7,7]
            nn.AdaptiveAvgPool2d(1)  # [512x1x1]
        )

        # 时间辅助分支（前两帧，共享空间降维+1D时间卷积）
        self.temporal_spatial_encoder = nn.Sequential(
            ResidualBlock(1, 32, stride=2, use_se=True, use_spatial=True),  # 224→112
            ResidualBlock(32, 64, stride=2, use_se=True, use_spatial=True),  # 112→56
            ResidualBlock(64, 128, stride=2, use_se=True),  # 56→28
            ResidualBlock(128, 256, stride=2, use_se=True),  # 28→14
            nn.AdaptiveAvgPool2d(1) # [256,14,14]
        )

        self.temporal_conv = nn.Sequential(
            # 输入：[B, 2, 256] → 展平空间维度后时间卷积
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=2),  # 时序特征提取
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1)  # 时间维度压缩为1 → [B, 128, 1]
        )

        # 特征融合（空间512 + 时间128 → 512）
        self.fusion_block = ResidualBlock(512 + 128, 1024, stride=1, use_se=True)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # 输入形状：[B, 3, 1, 224, 224] → 拆分三帧
        if x.shape[1] != 3:
            raise ValueError(f"Expected input with 3 frames, but got {x.shape[1]} frames.")
        img_t2, img_t1, img_t = x[:, 0], x[:, 1], x[:, 2]  # [B,1,224,224]

        # 空间主分支（第三帧）
        spatial_feat = self.spatial_branch(img_t)  # [B,512,1,1]
        spatial_feat = spatial_feat.flatten(1)  # [B,512]

        # 时间辅助分支（前两帧）
        feat_t2 = self.temporal_spatial_encoder(img_t2)  # [B,128,28,28]
        feat_t1 = self.temporal_spatial_encoder(img_t1)  # [B,128,28,28]
        # 拼接时间维度 → [B,2,256,1,1] → 展平空间 → [B,2,256,1,1]
        temporal_feat = torch.stack([feat_t2, feat_t1], dim=1).flatten(2)  # [B,2,256]
        # 1D时间卷积（通道维度在前）→ [B,256,1] → 展平 → [B,256]
        temporal_feat = self.temporal_conv(temporal_feat.permute(0, 2, 1)).squeeze(-1)

        # 特征融合（空间+时间）
        fused_feat = torch.cat([
            spatial_feat.unsqueeze(-1).unsqueeze(-1),  # [B,512,1,1]
            temporal_feat.unsqueeze(-1).unsqueeze(-1)  # [B,128,1,1]
        ], dim=1)  # [B,640,1,1]
        fused_feat = self.fusion_block(fused_feat).flatten(1)  # [B,512]

        return self.classifier(fused_feat)


# 训练
def train():
    data_root = r"C:\Users\xiang\OneDrive\桌面\subwayai\pythonProject\subwAI-surfer\traindata"
    batch_size = 16
    num_epochs = 100
    lr = 0.0005
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # plt.ion()
    # fig = plt.figure()
    # ax = fig.add_subplot(111)

    # 数据增强（对三帧应用相同变换）
    transform = transforms.Compose([
        # 随机裁剪
        TemporalTransform(transforms.RandomResizedCrop(224, scale=(0.8, 1.2))),
        # 随机模糊
        TemporalTransform(transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3)),
        # 随机改变亮度
        #TemporalTransform(transforms.RandomApply([transforms.ColorJitter(brightness=(0.9, 1.1))], p=0.3)),
        # 随机高斯噪声
        #TemporalTransform(transforms.RandomApply([AddGaussianNoise(std_range=(0.003, 0.01))], p=0.5)),
        #归一化
        #transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # 加载三帧数据集
    dataset = ImageDataset(root_dir=data_root, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size],
                                              generator=torch.Generator().manual_seed(42))

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
            {'params': decay_params, 'weight_decay': 1e-4},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]

    # 初始化模型
    model = TemporalSpatialModel(num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(get_optim_params(model), lr=lr, betas=(0.9, 0.999))

    # 学习率调度（warmup+余弦衰减）
    warmup_epochs = 10
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs),
            optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs)
        ],
        milestones=[warmup_epochs]
    )

    # 训练监控
    epoch_losses, epoch_val_losses = [], []
    epoch_accs, epoch_val_accs = [], []
    best_val_acc = -float('inf')
    patience = 8
    no_improve_epochs = 0
    best_model_weights = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)  # 输入形状：[B,3,1,224,224]

            # 验证输入形状
            if batch_idx == 0 and epoch == 0:
                print(f"输入形状: {images.shape}（[batch, T=3, C=1, H=224, W=224]）")

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 每5个batch打印进度
            if (batch_idx + 1) % 5 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}, Acc: {100 * correct / total:.2f}%')

        # 训练指标
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        epoch_losses.append(epoch_loss)
        epoch_accs.append(epoch_acc)

        # 验证阶段
        model.eval()
        val_loss_total, val_correct = 0.0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss_total += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_correct += (predicted == labels).sum().item()

        epoch_val_loss = val_loss_total / len(val_loader)
        epoch_val_losses.append(epoch_val_loss)
        val_acc = 100 * val_correct / len(val_dataset)
        epoch_val_accs.append(val_acc)

        # 学习率调度
        scheduler.step()

        # 打印日志
        print(f'\nEpoch {epoch + 1} 完成: Train Loss {epoch_loss:.4f}, Acc {epoch_acc:.2f}% | '
              f'Val Loss {epoch_val_loss:.4f}, Acc {val_acc:.2f}%')

        # 早停机制
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_weights = model.state_dict().copy()
            no_improve_epochs = 0
            print(f'验证准确率提升，保存最佳模型（当前最佳：{best_val_acc:.2f}%）')
        else:
            no_improve_epochs += 1
            print(f'验证准确率未提升，连续{no_improve_epochs}轮无改进（当前最佳：{best_val_acc:.2f}%）')

        if no_improve_epochs >= patience:
            print(f'\n早停触发！最佳验证准确率：{best_val_acc:.2f}%')
            break

        # # 绘制损失曲线
        # x = list(range(1, epoch + 2))
        # ax.clear()
        # ax.plot(x, epoch_losses, 'b-', label='训练损失')
        # ax.plot(x, epoch_val_losses, 'r-', label='验证损失')
        # ax.set_title('训练与验证损失曲线')
        # ax.set_xlabel('轮次')
        # ax.set_ylabel('损失值')
        # ax.legend()
        # fig.canvas.draw()
        # plt.pause(0.001)

    # 保存最佳模型
    if best_model_weights:
        torch.save(best_model_weights,
                   r"C:\Users\xiang\OneDrive\桌面\subwayai\pythonProject\subwAI-surfer\weights\temporal_spatial_model.pth")
        print(f'最佳模型已保存（验证准确率：{best_val_acc:.2f}%）')

    # plt.ioff()
    # plt.show()
    print('训练完成！')


if __name__ == '__main__':
    train()