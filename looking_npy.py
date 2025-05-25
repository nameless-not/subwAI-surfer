import numpy as np
import os
from pathlib import Path
from PIL import Image


def npy_to_png(npy_path: str, output_dir: str = "png_output/") -> None:
    """
    将三维张量灰度图像的 .npy 文件转换为 PNG 图像并保存

    Args:
        npy_path: 输入 .npy 文件路径（形状应为 (num_frames, height, width)）
        output_dir: 输出 PNG 图像的目录
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 加载张量
    tensor = np.load(npy_path)

    # 验证张量维度
    if tensor.ndim != 3:
        raise ValueError(f"输入张量应为三维，实际维度：{tensor.ndim}")

    num_frames, height, width = tensor.shape
    print(f"成功加载张量，尺寸：{num_frames} 帧 x {height}x{width}")

    # 遍历每一帧并保存为 PNG
    for frame_idx in range(num_frames):
        # 提取单帧图像
        frame = tensor[frame_idx]

        # 确保数据类型为 uint8，范围 0-255
        if frame.dtype != np.uint8:
            # 归一化到 0-255 范围
            frame_min, frame_max = frame.min(), frame.max()
            if frame_max == frame_min:
                # 处理所有值都相同的特殊情况
                frame_normalized = np.zeros_like(frame, dtype=np.uint8)
            else:
                frame_normalized = ((frame - frame_min) / (frame_max - frame_min) * 255).astype(np.uint8)
        else:
            frame_normalized = frame

        # 使用PIL保存为PNG
        img = Image.fromarray(frame_normalized)
        filename = Path(output_dir) / f"frame_{frame_idx + 1:03d}.png"
        img.save(str(filename))
        print(f"已保存帧 {frame_idx + 1}: {filename}")

    print(f"转换完成，共保存 {num_frames} 张 PNG 图像到 {output_dir}")


# ====================== 使用示例 ======================
if __name__ == "__main__":
    # 示例：处理单个 .npy 文件
    npy_path = r"C:\Users\xiang\OneDrive\桌面\subwayai\pythonProject\subwAI-surfer\traindata\3\tensor_right_20250525_164053_418025.npy"
    output_dir = r"C:\Users\xiang\OneDrive\桌面\subwayai\pythonProject\subwAI-surfer\traindata\png_output"

    npy_to_png(npy_path, output_dir)