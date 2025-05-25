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
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from functools import partial


# 数据加载
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, compute_stats=False):
        self.root_dir = root_dir
        self.transform = transform
        self.npy_paths = []  # 存储.npy文件路径
        self.labels = []  # 存储对应标签
        self.mean = 0.0
        self.std = 0.0
        self.scale_factor = 255

        for label in sorted(os.listdir(root_dir)):
            label_path = os.path.join(root_dir, label)
            if os.path.isdir(label_path) and label.isdigit():
                # 获取当前标签下所有.npy文件
                npy_files = glob.glob(os.path.join(label_path, "*.npy"))
                self.npy_paths.extend(npy_files)
                self.labels.extend([int(label)] * len(npy_files))

        self.labels = torch.tensor(self.labels, dtype=torch.long)

        # 首次加载时计算统计量
        if compute_stats:
            self._compute_stats()
    # 抽样计算均值和标准差
    def _compute_stats(self):

        total_pixels = 0
        sum_val = 0.0
        sum_sq = 0.0

        # 计算信息
        for npy_path in self.npy_paths:
            depth_images = np.load(npy_path).astype(np.float32)  # 原始数据：0-255
            depth_images_scaled = depth_images / self.scale_factor  # 缩放至0-1
            for img in depth_images_scaled:  # 遍历3个时间步的图像
                sum_val += img.sum()
                sum_sq += (img ** 2).sum()
                total_pixels += img.size  # 单张图像的像素总数

        # 计算缩放后的均值和标准差
        self.mean = sum_val / total_pixels
        self.std = np.sqrt((sum_sq / total_pixels) - (self.mean ** 2))
        print(f"缩放后数据统计完成：mean={self.mean:.4f}, std={self.std:.4f}")  

    def __len__(self):
        return len(self.npy_paths)

    def __getitem__(self, idx):
        npy_path = self.npy_paths[idx]
        depth_images = np.load(npy_path).astype(np.float32)
        depth_images_scaled = depth_images / self.scale_factor
        print(f"样本 {npy_path} 缩放后数值范围：min={depth_images_scaled.min()}, max={depth_images_scaled.max()}, mean={depth_images_scaled.mean()}")

        if depth_images_scaled.shape[0] != 3:
            raise ValueError(f"Invalid npy file {npy_path}: expected 3 time steps, got {depth_images_scaled.shape[0]}")

        # 转换为 (3, 1, H, W) 的 tensor（已缩放）
        image_tensor = torch.from_numpy(depth_images_scaled).unsqueeze(1)  # 形状 (3,1,H,W)

        if self.transform:
            image_tensor = self.transform(image_tensor)

        label = self.labels[idx]
        return image_tensor, label

# 随机噪声函数
def add_random_noise(tensor, mean, std):
    """
    添加随机噪声后使用均值标准差归一化
    :param tensor: 输入张量（数据范围应为原始数据范围，非归一化后）
    :param mean: 训练集全局均值
    :param std: 训练集全局标准差
    :return: 归一化后的含噪张量
    """
    # 生成0.001~0.003的随机噪声标准差
    noise_std = torch.rand(1, device=tensor.device) * (0.003 - 0.001) + 0.001
    noise = torch.randn(tensor.size(), device=tensor.device) * noise_std
    noisy_tensor = tensor + noise  # 添加噪声

    # 均值标准差归一化：(x - mean) / std
    normalized_tensor = (noisy_tensor - mean) / std
    return normalized_tensor

# 随机亮度和对比度
def random_illumination(tensor):
    """
    随机调整亮度和对比度
    :param tensor: 输入Tensor，形状 (C=1, H, W)
    :return: 调整后的Tensor，形状不变
    """
    # 随机亮度因子（0.8~1.2，可根据需求调整范围）
    brightness_factor = torch.rand(1, device=tensor.device) * 0.4 + 0.9  # [0.8, 1.2]
    tensor = tensor * brightness_factor  # 亮度调整

    # 随机对比度因子（0.8~1.2）
    contrast_factor = torch.rand(1, device=tensor.device) * 0.4 + 0.8  # [0.8, 1.2]
    mean = tensor.mean()  # 当前均值
    tensor = (tensor - mean) * contrast_factor + mean  # 对比度调整

    # 限制数据范围在[0,1]（避免溢出）
    return tensor.clamp(0.0, 1.0)


# 数据增强统一
class TemporalTransform:
    def __init__(self, transform):
        self.transform = transform
        # 记录需要共享参数的变换类型（如RandomResizedCrop、RandomHorizontalFlip）
        self.geometric_transforms = (transforms.RandomResizedCrop, transforms.RandomHorizontalFlip,
                                     transforms.RandomVerticalFlip, transforms.RandomRotation)

    def __call__(self, x):
        # x形状：(3, 1, H, W)（3个时间步，1通道，HxW尺寸）
        # 生成共享的随机参数（仅对几何变换）
        params = None
        for frame in x:
            # 仅需为第一个时间步生成参数，后续时间步复用
            if isinstance(self.transform, self.geometric_transforms):
                if isinstance(self.transform, transforms.RandomResizedCrop):
                    i, j, h, w = self.transform.get_params(
                        frame,
                        self.transform.scale,
                        self.transform.ratio
                    )
                    params = (i, j, h, w)
                elif isinstance(self.transform, transforms.RandomHorizontalFlip):
                    params = self.transform.get_params(frame)
                elif isinstance(self.transform, transforms.RandomVerticalFlip):
                    params = self.transform.get_params(frame)
                elif isinstance(self.transform, transforms.RandomRotation):
                    angle = self.transform.get_params(self.transform.degrees)
                    params = angle
            break

        # 对所有时间步应用相同变换
        transformed = []
        for frame in x:
            # 几何变换：使用共享参数
            if isinstance(self.transform, self.geometric_transforms) and params is not None:
                if isinstance(self.transform, transforms.RandomResizedCrop):
                    i, j, h, w = params
                    cropped_frame = F.crop(frame, i, j, h, w)
                    transformed_frame = F.resize(cropped_frame, self.transform.size)
                elif isinstance(self.transform, transforms.RandomHorizontalFlip):
                    transformed_frame = F.hflip(frame) if params else frame
                elif isinstance(self.transform, transforms.RandomVerticalFlip):
                    transformed_frame = F.vflip(frame) if params else frame
                elif isinstance(self.transform, transforms.RandomRotation):
                    angle = params
                    transformed_frame = F.rotate(frame, angle)
            else:
                # 非几何变换（如亮度、噪声）：独立应用
                transformed_frame = self.transform(frame)

            # 确保输出为Tensor（兼容PIL变换）
            if isinstance(transformed_frame, Image.Image):
                transformed_frame = transforms.ToTensor()(transformed_frame)

            transformed.append(transformed_frame.unsqueeze(0))  # 恢复时间步维度
        return torch.cat(transformed, dim=0)  # 合并时间步，形状(3, 1, 224, 224)

# 跨模态融合
class CrossModalFusion(nn.Module):
    def __init__(self, spatial_dim, temporal_dim, fusion_dim):
        super().__init__()
        self.spatial_proj = nn.Linear(spatial_dim, fusion_dim)
        self.temporal_proj = nn.Linear(temporal_dim, fusion_dim)
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.Sigmoid()
        )
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.SiLU()
        )

    def forward(self, spatial_feat, temporal_feat):
        """
        输入:
            spatial_feat: [B, spatial_dim] 空间特征
            temporal_feat: [B, temporal_dim] 时间特征
        输出:
            fused_feat: [B, fusion_dim] 融合特征
        """
        s = self.spatial_proj(spatial_feat)  # [B, fusion_dim]
        t = self.temporal_proj(temporal_feat)  # [B, fusion_dim]
        gate = self.gate(torch.cat([s, t], dim=-1))  # [B, fusion_dim]
        fused = gate * s + (1 - gate) * t  # 门控融合
        return self.fusion(fused)

# 通道注意力
class ECAAttention(nn.Module):
    def __init__(self, kernel_size=3):  # 经验核大小3/5/7
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化到[B,C,1,1]
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size,
                             padding=kernel_size//2, bias=False)  # 1D卷积捕获局部通道依赖
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C = x.shape[:2]
        # 池化后调整维度：[B,C,1,1] → [B,1,C]（适配1D卷积）
        y = self.avg_pool(x).view(B, C, 1).transpose(1, 2)
        y = self.conv(y)  # [B,1,C]
        # 恢复维度并应用sigmoid：[B,C,1,1]
        y = y.transpose(1, 2).view(B, C, 1, 1)
        return x * self.sigmoid(y)

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
                 use_eca=True, use_spatial=True, eca_kernel=3, spatial_kernel=7, dropout_rate=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.act = nn.SiLU()

        # 输入通道>=128时触发dropout
        self.dropout = nn.Dropout(dropout_rate) if in_channels >= 128 else None

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(num_groups=8, num_channels=out_channels)

        attn_layers = []

        # 选择是否应用通道/空间注意力机制
        self.use_eca = use_eca
        self.use_spatial = use_spatial
        if use_eca:
            self.eca = ECAAttention(eca_kernel)
        if use_spatial:
            self.spatial_attn = SpatialAttention(kernel_size=spatial_kernel)

        # 注意力融合
        fusion_in=out_channels * (2 if (use_eca and use_spatial) else 1)
        self.fusion = nn.Conv2d(fusion_in, out_channels, 1) if (use_eca or use_spatial) else nn.Identity()

        self.residual = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.GroupNorm(num_groups=8, num_channels=out_channels)
            )

    def forward(self, x):
        residual = self.residual(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)

        # dropout
        if self.dropout is not None:
            x = self.dropout(x)

        x = self.conv2(x)
        x = self.bn2(x)

        # 并行注意力处理
        attn_feats = []
        if self.use_eca:
            attn_feats.append(self.eca(x))
        if self.use_spatial:
            attn_feats.append(self.spatial_attn(x))

        # 融合注意力特征
        if attn_feats:
            if len(attn_feats) == 1:
                x = attn_feats[0]  # 仅一个注意力模块
            else:
                x = torch.cat(attn_feats, dim=1)  # 拼接通道（[B,2C,H,W]）
                x = self.fusion(x)  # 1x1卷积融合回[B,C,H,W]

        x += residual
        return self.act(x)


# 1+2D时序空间模型
class TemporalSpatialModel(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        # 空间主分支（处理当前帧）
        self.spatial_branch = nn.Sequential(
            ResidualBlock(1, 32, stride=2, use_eca=True, use_spatial=True, spatial_kernel=9),    # [32, 112, 112]
            ResidualBlock(32, 64, stride=2, use_eca=True, use_spatial=True, spatial_kernel=7),             # [64, 56, 56]
            ResidualBlock(64, 128, stride=2, use_eca=True, use_spatial=True, spatial_kernel=5),  # [128, 28, 28]
            ResidualBlock(128, 256, stride=2, use_eca=True, use_spatial=True, spatial_kernel=3),                # [256, 14, 14]
            ResidualBlock(256, 512, stride=2, use_eca=True),                # [512, 7, 7]
            nn.AdaptiveAvgPool2d(1)                                      # [512, 1, 1]
        )

        # 时间特征编码器（处理历史帧）
        self.temporal_encoder = nn.Sequential(
            ResidualBlock(1, 32, stride=2, use_eca=True, use_spatial=True, spatial_kernel=7),  # 224→112
            ResidualBlock(32, 64, stride=2, use_eca=True, use_spatial=True),  # 112→56
            ResidualBlock(64, 128, stride=2, use_eca=True),                 # 56→28
            ResidualBlock(128,256, stride=2, use_eca=True),               # 28→14
            nn.AdaptiveAvgPool2d(1)                                      # [256, 1, 1]
        )

        # LSTM
        self.temporal_lstm = nn.LSTM(
            input_size=256,  # 单帧特征维度
            hidden_size=128,  # 隐藏层维度
            num_layers=2,  # 2层LSTM
            batch_first=True,  # 输入形状[B,T,D]
            bidirectional=True  # 双向LSTM捕捉前后依赖
        )

        # LSTM输出映射到融合维度
        self.lstm_proj = nn.Linear(128 * 2, 192)  # 双向输出维度=2*hidden_size

        # 跨模态融合模块（空间512 + 时间192 → 512）
        self.cross_fusion = CrossModalFusion(
            spatial_dim=512,
            temporal_dim=192,
            fusion_dim=512
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # 输入形状: [B, 3, 1, 224, 224]（3个时间步）
        if x.shape[1] != 3:
            raise ValueError(f"需要3个时间步输入，当前输入{ x.shape[1] }个")

        # 拆分时间步（t0: 最早帧, t1: 中间帧, t2: 当前帧）
        t0, t1, t2 = x[:, 0], x[:, 1], x[:, 2]  # [B,1,224,224]

        # 空间主分支（处理当前帧t2）
        spatial_feat = self.spatial_branch(t2).flatten(1)  # [B,512]

        # 时间分支（处理历史帧t0/t1/t2）
        t0_feat = self.temporal_encoder(t0).flatten(1)  # [B,256]
        t1_feat = self.temporal_encoder(t1).flatten(1)  # [B,256]
        t2_feat = self.temporal_encoder(t2).flatten(1)  # [B,256]
        temporal_seq = torch.stack([t0_feat, t1_feat, t2_feat], dim=1)  # [B,3,256]

        # LSTM处理时序序列
        lstm_out, _ = self.temporal_lstm(temporal_seq)  # [B,3,256] → [B,3,256]（双向LSTM输出）
        lstm_feat = lstm_out.mean(dim=1)  # 时间步维度取平均 → [B,256]
        temporal_feat = self.lstm_proj(lstm_feat)  # 映射到192维

        # 跨模态融合
        fused_feat = self.cross_fusion(spatial_feat, temporal_feat)

        return self.classifier(fused_feat)



# 定义具名函数(用于数据增强中pickle序列化)
def apply_random_illumination(x):
    return random_illumination(x)
def apply_random_noise(x, mean, std):
    return add_random_noise(x, mean=mean, std=std)

# 训练
def train():
    data_root = r"C:\Users\xiang\OneDrive\桌面\subwayai\pythonProject\subwAI-surfer\traindata"
    batch_size = 16
    num_epochs = 150
    lr = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # # 初始化混合精度组件
    # scaler = GradScaler()  # 梯度缩放器

    # plt.ion()
    # fig = plt.figure()
    # ax = fig.add_subplot(111)

    # 创建临时Dataset（不应用transform，仅计算统计量）
    temp_dataset = ImageDataset(root_dir=data_root, transform=None, compute_stats=True)
    data_mean = temp_dataset.mean  # 获取统计的均值
    data_std = temp_dataset.std  # 获取统计的标准差

    # 数据增强（对三帧应用相同变换）
    transform = transforms.Compose([
        TemporalTransform(transforms.RandomResizedCrop(224, scale=(0.93, 1.07))),  # 随机裁剪
        TemporalTransform(transforms.RandomApply([transforms.Lambda(apply_random_illumination)], p=0.3)),  # 随机亮度
        TemporalTransform(transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2)),  # 随机模糊
        TemporalTransform(transforms.RandomApply([  # 随机擦除
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.05), ratio=(0.5, 2.0))
        ], p=0.3)),
        TemporalTransform(transforms.RandomApply([  # 随机噪声
            transforms.Lambda(partial(apply_random_noise, mean=data_mean, std=data_std))
        ], p=0.3)),
        TemporalTransform(transforms.Normalize(mean=[data_mean], std=[data_std])),  # 使用实际统计值归一化
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
            {'params': decay_params, 'weight_decay': 3e-4},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]

    # 初始化模型
    model = TemporalSpatialModel(num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(get_optim_params(model), lr=lr, betas=(0.9, 0.999))

    # 学习率调度（warmup+余弦衰减）
    warmup_epochs = 5
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs),  # 线性warmup
            CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)  # 余弦退火重启
        ],
        milestones=[warmup_epochs]  # warmup结束后切换
    )

    # 训练监控
    epoch_losses, epoch_val_losses = [], []
    epoch_accs, epoch_val_accs = [], []
    best_val_acc = -float('inf')
    patience = 10
    no_improve_epochs = 0
    best_model_weights = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)  # 输入形状：[B,3,1,224,224]

            # # 混合精度前向传播
            # with autocast():  # 自动混合精度上下文
            #     outputs = model(images)
            #     loss = criterion(outputs, labels)
            #
            # # 混合精度反向传播
            # optimizer.zero_grad()
            # scaler.scale(loss).backward()  # 缩放损失后反向传播
            # scaler.step(optimizer)  # 缩放优化器步骤
            # scaler.update()  # 更新缩放因子

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
            print(f'\n早停触发！')
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