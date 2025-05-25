import mss
import cv2
import numpy as np
import time
import keyboard
import os
from datetime import datetime


class EnhancedScreenCapture:
    def __init__(self, region=None, output_path="screenshots/"):
        """初始化屏幕截图工具"""
        self.region = region or self.get_fullscreen_region()
        self.output_path = output_path
        self.frame = None
        self.last_time = time.time()
        self.auto_interval = 0.05  # 自动截图间隔（0.05秒）
        self.auto_screenshots = []  # 自动截图缓存（最多保存6张）
        self.max_cache_size = 6  # 增加缓存大小以支持更多历史帧

        # 定义保存目录
        self.directories = {
            'up': os.path.join(output_path, '4'),
            'down': os.path.join(output_path, '1'),
            'left': os.path.join(output_path, '2'),
            'right': os.path.join(output_path, '3'),
            'auto': os.path.join(output_path, '0')
        }

        # 创建所有目录
        for dir_path in self.directories.values():
            os.makedirs(dir_path, exist_ok=True)

        # 按键映射
        self.key_mapping = {'up': 'up', 'down': 'down', 'left': 'left', 'right': 'right'}
        self.key_states = {key: False for key in self.key_mapping}

        # 自动截图计时
        self.last_auto_save = time.time()
        self.last_key_press_time = time.time()

        # 增强功能计时器和状态
        self.enhanced_mode_active = False  # 增强模式激活状态
        self.enhanced_mode_start_time = 0  # 增强模式开始时间
        self.last_enhanced_check_time = 0  # 上次增强检查时间
        self.enhanced_direction = None  # 增强模式方向
        self.enhanced_check_intervals = [0.3, 0.6, 0.9]  # 检查时间点

        # 方向镜像映射
        self.mirror_mapping = {
            'up': 'up',
            'down': 'down',
            'left': 'right',
            'right': 'left',
            'auto': 'auto'
        }

    def get_fullscreen_region(self):
        """自动获取全屏区域"""
        with mss.mss() as sct:
            return sct.monitors[0]  # 全屏幕区域

    def capture_frame(self):
        """捕获当前屏幕帧并转换为灰度图"""
        with mss.mss() as sct:
            img = sct.grab(self.region)
            self.frame = cv2.cvtColor(np.array(img), cv2.COLOR_BGRA2GRAY)
            return self.frame

    def cache_auto_screenshot(self):
        """仅缓存自动截图，不保存到文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.auto_screenshots.append({
            'frame': self.frame.copy(),
            'timestamp': timestamp
        })
        # 保持缓存大小
        if len(self.auto_screenshots) > self.max_cache_size:
            self.auto_screenshots.pop(0)
        return timestamp

    def create_composite_image(self, direction):
        """创建合成图（仅保存原始张量NPY）"""
        if len(self.auto_screenshots) < 3:
            print("缓存不足，至少需要3张自动截图")
            return None, None, None, None, None, None

        # 提取缓存中的截图 - 取当前帧和前两帧
        img1 = self.auto_screenshots[-3]['frame']
        img2 = self.auto_screenshots[-2]['frame']
        img3 = self.auto_screenshots[-1]['frame']

        h, w = img3.shape[:2]
        img1 = cv2.resize(img1, (w, h))
        img2 = cv2.resize(img2, (w, h))

        # 生成原始张量 [3, H, W]
        composite_tensor = np.stack([img1, img2, img3], axis=0)

        # 获取镜像方向
        mirror_direction = self.mirror_mapping[direction]

        # 生成镜像张量
        mirrored_tensor = np.flip(composite_tensor, axis=2).copy()

        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        # 保存原始张量
        dir_path = self.directories[direction]
        tensor_filename = os.path.join(dir_path, f"tensor_{direction}_{timestamp}.npy")
        np.save(tensor_filename, composite_tensor)

        # 保存镜像张量
        mirror_dir_path = self.directories[mirror_direction]
        mirror_tensor_filename = os.path.join(mirror_dir_path, f"tensor_{mirror_direction}_{timestamp}_mirror.npy")
        np.save(mirror_tensor_filename, mirrored_tensor)

        print(f"原始张量保存: {tensor_filename}")
        print(f"镜像张量保存: {mirror_tensor_filename}")

        return composite_tensor, None, tensor_filename, mirrored_tensor, None, mirror_tensor_filename

    def handle_enhanced_capture(self):
        """处理增强模式下的截图逻辑"""
        current_time = time.time()
        elapsed_time = current_time - self.enhanced_mode_start_time

        # 检查是否超过1秒
        if elapsed_time > 1.0:
            self.enhanced_mode_active = False
            return

        # 检查是否需要在特定时间点执行截图
        for interval in self.enhanced_check_intervals:
            if (interval - 0.005) <= elapsed_time < (interval + 0.005):
                # 检查是否有按键操作
                any_key_pressed = any(keyboard.is_pressed(key) for key in self.key_mapping)
                if not any_key_pressed:
                    # 没有按键操作，执行截图
                    print(f"增强模式: 在按键后{interval:.1f}秒检测到无按键，保存截图")
                    self.create_composite_image('auto')
                else:
                    # 有按键操作，重置增强模式
                    self.enhanced_mode_start_time = current_time
                    print("增强模式: 检测到新按键，重置计时器")
                break

    def run(self):
        """启动程序主循环（无窗口显示）"""
        print("程序启动 - 功能说明:")
        print("  自动模式: 每0.05秒缓存截图（内存中，最多6张）")
        print("  按键模式: W/S/A/D 生成合成图和镜像图")
        print("  无操作模式: 1秒无按键自动生成合成图和镜像图")
        print("  增强模式: 按键操作后1秒内，每0.3秒检测无按键则保存截图")
        print("按 ESC 退出程序")

        while not keyboard.is_pressed('esc'):
            self.capture_frame()

            # 处理自动缓存
            if time.time() - self.last_auto_save >= self.auto_interval:
                self.cache_auto_screenshot()
                self.last_auto_save = time.time()

            # 检测按键操作
            key_pressed = False
            for key, direction in self.key_mapping.items():
                if keyboard.is_pressed(key) and not self.key_states[key]:
                    self.key_states[key] = True
                    self.last_key_press_time = time.time()
                    key_pressed = True
                    # 生成合成图和镜像图
                    self.create_composite_image(direction)

                    # 激活增强模式
                    self.enhanced_mode_active = True
                    self.enhanced_mode_start_time = time.time()
                    self.enhanced_direction = direction
                    print(f"激活增强模式: {direction}")

            # 重置按键状态
            for key in self.key_mapping:
                if not keyboard.is_pressed(key):
                    self.key_states[key] = False

            # 处理增强模式
            if self.enhanced_mode_active:
                self.handle_enhanced_capture()

            # 检测无按键操作时间
            idle_time = time.time() - self.last_key_press_time
            if idle_time >= 1 and len(self.auto_screenshots) >= 4:
                self.last_key_press_time = time.time()
                print("检测到1秒无按键操作，自动生成合成图")
                self.create_composite_image('auto')

            time.sleep(0.001)  # 降低CPU占用

        print("程序退出")

    @staticmethod
    def load_composite_tensor(file_path):
        """加载原始张量"""
        try:
            tensor = np.load(file_path)
            print(f"成功加载张量，维度: {tensor.shape}")
            return tensor
        except Exception as e:
            print(f"加载失败: {e}")
            return None


# ====================== 使用示例 ======================
if __name__ == "__main__":
    # 自定义截图区域（示例：全屏）
    custom_region = {
        "top": 0,
        "left": 0,
        "width": 800,  # 屏幕宽度
        "height": 600  # 屏幕高度
    }

    # 创建实例
    sc = EnhancedScreenCapture(
        region=custom_region,
        output_path=r"C:\Users\xiang\OneDrive\桌面\subwayai\pythonProject\subwAI-surfer\savenpy"
    )

    # 启动程序（无窗口显示）
    sc.run()