import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import glob
import matplotlib.pyplot as plt
from torch.utils.data import random_split
import numpy as np
import torch.nn.functional as F


# 数据加载
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = r"C:\Users\xiang\OneDrive\桌面\subwayai\pythonProject\subwAI-surfer\data"
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
        label = self.labels[idx]

        # 加载.npy文件[深度, 高, 宽]
        depth_images = np.load(npy_path)  # 输出形状：(3, H, W)

        # 只保留第三张图片（索引2）
        third_image = depth_images[2:3]  # 形状变为 (1, H, W)

        # 归一化
        if third_image.dtype == np.uint8:
            third_image = third_image.astype(np.float32) / 255.0

        image_tensor = torch.from_numpy(third_image)  # 形状：(1, H, W)

        if self.transform:
            image_tensor = self.transform(image_tensor)

        return image_tensor, label


# 深度可分离卷积（Depthwise + Pointwise）
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=stride,
            padding=1, groups=in_channels, bias=False
        )  # 深度卷积（每个通道单独卷积）
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, bias=False
        )  # 逐点卷积（跨通道融合）
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU6()  # 使用ReLU6保持轻量化

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


# 逆残差块（先升维再降维）
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion=2):
        super().__init__()
        self.stride = stride
        hidden_dim = int(in_channels * expansion)  # 扩展维度

        # 1x1 升维卷积
        self.expand = nn.Conv2d(in_channels, hidden_dim, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)

        # 深度可分离卷积（保持维度）
        self.dw_conv = DepthwiseSeparableConv(hidden_dim, hidden_dim, stride=stride)

        # 1x1 降维卷积（线性激活，无ReLU）
        self.reduce = nn.Conv2d(hidden_dim, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 跳跃连接（仅当输入输出形状相同时启用）
        self.use_res_connect = (stride == 1 and in_channels == out_channels)

    def forward(self, x):
        identity = x
        x = self.expand(x)
        x = self.bn1(x)
        x = F.relu6(x)
        x = self.dw_conv(x)
        x = self.reduce(x)
        x = self.bn2(x)
        if self.use_res_connect:
            return x + identity
        return x


# 定义完整模型
class LightweightModel(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        # 输入为单通道（第三张图片）
        self.first_conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6()
        )

        # 逆残差块堆叠
        self.blocks = nn.Sequential(
            InvertedResidual(16, 32, stride=2, expansion=4),  # 输出尺寸减半
            InvertedResidual(32, 32, stride=1, expansion=4),  # 保持尺寸
            InvertedResidual(32, 64, stride=2, expansion=4),  # 输出尺寸减半
            InvertedResidual(64, 64, stride=1, expansion=4),
            InvertedResidual(64, 128, stride=2, expansion=4),  # 输出尺寸减半

            nn.Conv2d(128,512,1)
            nn.ReLU6()
        )

        # 全局平均池化 + 分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 全局平均池化到1x1
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU6(),
            nn.Dropout1d(0.2)
            nn.Linear(128, num_classes)  # 输出类别数
        )

    def forward(self, x):
        x = self.first_conv(x)
        x = self.blocks(x)
        x = self.classifier(x)
        return x


# 训练
def train():
    data_root = r"C:\Users\xiang\OneDrive\桌面\subwayai\pythonProject\subwAI-surfer\data"
    batch_size = 16
    num_epochs = 100
    lr = 0.0005
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # 数据预处理
    transform = transforms.Compose([
        transforms.Normalize(  # 归一化
            mean=[0.5],
            std=[0.5]
        )
    ])

    # 加载数据集
    dataset = ImageDataset(root_dir=data_root, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size],
                                              generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    def get_optim_params(model):
        param_groups = []
        # 收集需要权重衰减的参数（通常是卷积/全连接的权重）
        decay_params = []
        # 收集不需要权重衰减的参数（偏置、归一化层参数）
        no_decay_params = []

        for name, param in model.named_parameters():
            # 跳过未训练的参数
            if not param.requires_grad:
                continue
            # 识别偏置项（名称包含'bias'）
            if 'bias' in name:
                no_decay_params.append(param)
            # 识别归一化层（名称包含'bn'或'norm'）
            elif 'bn' in name or 'norm' in name:
                no_decay_params.append(param)
            # 其他参数（主要是卷积/全连接权重）
            else:
                decay_params.append(param)

        return [
            {'params': decay_params, 'weight_decay': 5e-5},  # 权重衰减组
            {'params': no_decay_params, 'weight_decay': 0.0}  # 无权重衰减组
        ]

    # 初始化模型
    model = LightweightModel(num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        get_optim_params(model),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        amsgrad=False
    )
    max_lr = 0.001  # 最大学习率
    total_steps = num_epochs * len(train_loader)  # 总步数
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, total_steps=total_steps)

    epoch_losses = []
    epoch_val_losses = []
    epoch_accs = []
    epoch_val_accs = []

    best_val_loss = float('inf')
    best_val_acc = -float('inf')
    patience = 10
    no_improve_epochs = 0
    no_improve_acc_epochs = 0
    best_model_weights = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # 验证输入形状
            if batch_idx == 0 and epoch == 0:
                print(f"输入形状: {images.shape}（[batch, channel, H, W]）")  # 应输出 [batch_size, 1, H, W]

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if (batch_idx + 1) % 5 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}, Acc: {100 * correct / total:.2f}%')

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        epoch_accs.append(epoch_acc)
        epoch_losses.append(epoch_loss)

        print(f'\nEpoch [{epoch + 1}/{num_epochs}] Complete: '
              f'Avg Loss: {epoch_loss:.4f}, Avg Acc: {epoch_acc:.2f}%\n')

        # 验证损失
        model.eval()
        val_loss_total = 0.0
        val_correct = 0
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
        scheduler.step(epoch_val_loss)

        val_acc = 100 * val_correct / len(val_dataset)
        epoch_val_accs.append(val_acc)
        print(f'Validation Acc: {val_acc:.2f}%, Val Loss: {epoch_val_loss:.4f}')

        # 早停机制
        # if epoch_val_loss < best_val_loss:
        #     best_val_loss = epoch_val_loss
        #     no_improve_epochs = 0
        # else:
        #     no_improve_epochs += 1

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_weights = model.state_dict().copy()
            no_improve_acc_epochs = 0
        else:
            no_improve_acc_epochs += 1

        if no_improve_epochs >= patience or no_improve_acc_epochs >= patience:
            print(f'\nEarly stopping triggered at epoch {epoch + 1}!')
            model.load_state_dict(best_model_weights)
            torch.save(best_model_weights,
                       r"C:\Users\xiang\OneDrive\桌面\subwayai\pythonProject\subwAI-surfer\weights\temp0model.pth")
            break

        print(f'Best validation accuracy: {best_val_acc:.2f}')

        # 绘制损失曲线
        x = list(range(1, epoch + 2))
        ax.clear()
        ax.plot(x, epoch_losses, 'b-', label='Train Loss')
        ax.plot(x, epoch_val_losses, 'r-', label='Val Loss')
        ax.set_title('Training & Validation Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.001)

    plt.ioff()
    plt.show()

    print('训练完成！')


if __name__ == '__main__':
    train()