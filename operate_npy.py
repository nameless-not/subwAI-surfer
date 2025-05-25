import numpy as np
import cv2
import os
import argparse
import json
from pathlib import Path
from tqdm import tqdm  # 用于显示进度条


class InplaceNpyProcessor:
    """在同一文件夹内处理 .npy 文件"""

    def __init__(self, input_folders: list, target_size: tuple = (320, 240),
                 suffix: str = "_resized", processed_log: str = 'processed_files.json'):
        """
        初始化处理器3

        Args:
            input_folders: 输入文件夹列表
            target_size: 目标图像尺寸 (宽, 高)
            suffix: 处理后文件的后缀，设为空字符串可覆盖原文件
            processed_log: 已处理文件记录的 JSON 文件路径
        """
        self.input_folders = input_folders
        self.target_size = target_size
        self.suffix = suffix
        self.processed_log = processed_log
        self.processed_files = self._load_processed_log()

        # 新增：验证目标尺寸是否有效
        if not all(isinstance(dim, int) and dim > 0 for dim in target_size):
            raise ValueError(f"无效的目标尺寸: {target_size}，必须为正整数元组")

    def _load_processed_log(self) -> set:
        """加载已处理文件记录"""
        if os.path.exists(self.processed_log):
            try:
                with open(self.processed_log, 'r') as f:
                    return set(json.load(f))
            except Exception as e:
                print(f"警告: 加载处理记录失败: {e}，将创建新记录")
        return set()

    def _save_processed_log(self) -> None:
        """保存已处理文件记录"""
        try:
            with open(self.processed_log, 'w') as f:
                json.dump(list(self.processed_files), f)
        except Exception as e:
            print(f"警告: 保存处理记录失败: {e}")

    def _is_processed(self, file_path: str) -> bool:
        """检查文件是否已处理"""
        return file_path in self.processed_files

    def _mark_processed(self, file_path: str) -> None:
        """标记文件为已处理"""
        self.processed_files.add(file_path)
        self._save_processed_log()

    def _get_output_path(self, input_path: str) -> str:
        """获取输出文件路径"""
        if not self.suffix:
            return input_path  # 覆盖原文件

        dirname, basename = os.path.split(input_path)
        root, ext = os.path.splitext(basename)
        return os.path.join(dirname, f"{root}{self.suffix}{ext}")

    def _unsharp_mask(self, image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
        """非锐化掩蔽增强版"""
        blurred = cv2.GaussianBlur(image, kernel_size, sigma)
        sharpened = float(amount + 1) * image - float(amount) * blurred
        sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
        sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
        sharpened = sharpened.round().astype(np.uint8)
        if threshold > 0:
            low_contrast_mask = np.absolute(image - blurred) < threshold
            np.copyto(sharpened, image, where=low_contrast_mask)
        return sharpened

    def _process_npy_file(self, input_path: str) -> bool:
        """处理单个 .npy 文件"""
        try:
            # 检查文件是否已处理
            if self._is_processed(input_path):
                print(f"跳过已处理文件: {input_path}")
                return True

            # 加载数据
            data = np.load(input_path)

            # 验证数据形状
            if data.ndim != 3:
                print(f"警告: 文件 '{input_path}' 不是三维张量，跳过")
                return False

            # 新增：打印原始尺寸
            original_size = (data.shape[2], data.shape[1])  # (宽, 高)
            print(f"处理前尺寸: {original_size}")

            # 获取输出路径
            output_path = self._get_output_path(input_path)

            # 调整所有图像大小
            resized_data = np.zeros((data.shape[0], self.target_size[1], self.target_size[0]),
                                    dtype=data.dtype)

            for i in range(data.shape[0]):
                # 使用Lanczos-2插值调整单张图像
                resized_data[i] = cv2.resize(data[i], self.target_size,
                                             interpolation=cv2.INTER_LANCZOS4)
                # 应用非锐化掩蔽增强版进行锐化
                resized_data[i] = self._unsharp_mask(resized_data[i])

            # 保存结果（如果后缀为空则覆盖原文件）
            np.save(output_path, resized_data)

            # 新增：验证处理后的尺寸
            if os.path.exists(output_path):
                try:
                    test_data = np.load(output_path)
                    processed_size = (test_data.shape[2], test_data.shape[1])
                    print(f"处理后尺寸: {processed_size}")

                    if processed_size != self.target_size:
                        print(f"警告: 处理后尺寸不匹配! 预期: {self.target_size}，实际: {processed_size}")
                except Exception as e:
                    print(f"警告: 无法验证处理后文件尺寸: {e}")

            if self.suffix:
                print(f"已处理: {input_path} -> {output_path}")
            else:
                print(f"已处理并覆盖: {input_path}")

            # 标记为已处理
            self._mark_processed(input_path)
            return True

        except Exception as e:
            print(f"错误: 处理文件 '{input_path}' 时发生异常: {e}")
            return False

    def process(self) -> None:
        """处理所有文件夹中的 .npy 文件"""
        # 收集所有 .npy 文件
        npy_files = []

        for folder in self.input_folders:
            if not os.path.exists(folder):
                print(f"警告: 输入文件夹 '{folder}' 不存在，跳过")
                continue

            print(f"扫描文件夹: {folder}")
            for root, _, files in os.walk(folder):
                for file in files:
                    if file.lower().endswith('.npy'):
                        npy_files.append(os.path.join(root, file))

        print(f"找到 {len(npy_files)} 个 .npy 文件")

        # 处理所有文件
        success_count = 0
        failed_count = 0

        for file_path in tqdm(npy_files, desc="处理进度"):
            if self._process_npy_file(file_path):
                success_count += 1
            else:
                failed_count += 1

        print(f"处理完成! 成功: {success_count}, 失败: {failed_count}, 已跳过: {len(self.processed_files)}")


def main():
    """主函数：解析命令行参数并执行处理"""
    # 方式一：通过命令行参数传递（保留原有功能）
    parser = argparse.ArgumentParser(description='在同一文件夹内处理 .npy 文件')
    parser.add_argument('-i', '--input', nargs='+', required=False,  # 修改为可选参数
                        help='输入文件夹路径列表，用空格分隔')
    parser.add_argument('-s', '--size', type=int, nargs=2, default=(320, 240),
                        metavar=('WIDTH', 'HEIGHT'),
                        help='目标图像尺寸 (宽, 高)，默认为 320x240')
    parser.add_argument('--suffix', default='_resized',
                        help='处理后文件的后缀，设为空字符串可覆盖原文件')
    parser.add_argument('-l', '--log', default='processed_files.json',
                        help='已处理文件记录的 JSON 文件路径')
    parser.add_argument('--force', action='store_true',
                        help='强制重新处理所有文件，忽略处理记录')

    args = parser.parse_args()

    # 方式二：直接在代码中设置参数值（新增功能）
    if args.input:
        # 使用命令行参数
        processor = InplaceNpyProcessor(
            input_folders=args.input,
            target_size=tuple(args.size),
            suffix=args.suffix,
            processed_log=args.log
        )
    else:
        # 使用代码中硬编码的参数
        processor = InplaceNpyProcessor(
            input_folders=[
                r"C:\Users\xiang\OneDrive\桌面\subwayai\pythonProject\subwAI-surfer\traindata\0",
                r"C:\Users\xiang\OneDrive\桌面\subwayai\pythonProject\subwAI-surfer\traindata\1",
                r"C:\Users\xiang\OneDrive\桌面\subwayai\pythonProject\subwAI-surfer\traindata\2",
                r"C:\Users\xiang\OneDrive\桌面\subwayai\pythonProject\subwAI-surfer\traindata\3",
                r"C:\Users\xiang\OneDrive\桌面\subwayai\pythonProject\subwAI-surfer\traindata\4"
            ],
            target_size=(320, 240),  # 设置为你需要的尺寸[W,H]
            suffix="",  # 设为空字符串表示覆盖原文件
            processed_log='processed_files.json'
        )

    # 新增：强制重新处理选项
    if args.force:
        print("警告: 强制重新处理所有文件，将忽略已有的处理记录")
        processor.processed_files = set()

    processor.process()


if __name__ == "__main__":
    main()