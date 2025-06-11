import mss
import cv2
import numpy as np
import time
import keyboard
import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from pynput.keyboard import Controller, Key
from threading import Thread
from queue import Queue
from datetime import datetime
from Models.sl_worker_new import TemporalSpatialModel, TemporalTransform, ImageDataset  # 导入模型、TemporalTransform 和 ImageDataset

time.sleep(1)

# 非锐化掩蔽增强版
def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

# ====================== 游戏控制类 ======================
class GameController:
    def __init__(self):
        self.keyboard = Controller()
        # 动作映射
        self.action_map = {
            0: None,    # 无操作
            1: Key.down,  # 下方向键
            2: Key.left,  # 左方向键
            3: Key.right, # 右方向键
            4: Key.up     # 上方向键
        }
        self.last_action_time = -0.5

    # 模拟键盘操作
    def perform_action(self, action):
        key = self.action_map.get(action, None)
        if key:  # 仅处理非0动作
            current_time = time.time()
            # 检查缓冲期
            if current_time - self.last_action_time < 0.4:
                print(f"缓冲期（剩余{0.4 - (current_time - self.last_action_time):.2f}秒），跳过动作: {key}")
                return

            # 执行按键操作
            try:
                self.keyboard.press(key)
                time.sleep(0.02)  # 按键持续时间
                self.keyboard.release(key)
                print(f"执行键盘动作: {key}")
            except Exception as e:
                print(f"按键操作出错: {e}")

            # 更新时间戳
            self.last_action_time = current_time

# 实时推理控制器
class RealTimeInference:
    def __init__(self, model_path, region=None, target_size=(224, 224)):
        self.region = region or {"top": 0, "left": 0, "width": 800, "height": 600}  # 默认区域（可自定义）
        self.target_size = target_size
        self.auto_interval = 0.05
        self.max_cache = 3
        self.frame_cache = Queue(maxsize=self.max_cache)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(model_path)
        self.controller = GameController()

        self.transform = transforms.Compose([
            #TemporalTransform(transforms.Normalize(mean=[data_mean], std=[data_std])),
            transforms.Lambda(lambda x: F.interpolate(x, size=target_size, mode='bilinear', align_corners=False))
            # 调整尺寸
        ])

    def load_model(self, model_path):
        """加载训练好的模型"""
        model = TemporalSpatialModel(num_classes=5)
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()  # 推理模式
        return model.to(self.device)

    # 实时捕获屏幕帧并预处理
    def capture_frame(self):
        with mss.mss() as sct:
            img = sct.grab(self.region)
            # 转换为灰度图
            frame = np.array(img, dtype=np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
            # 用lanczo2下采样到320*240
            frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_LANCZOS4)
            # 用非锐化掩蔽增强版进行锐化
            frame = unsharp_mask(frame)
            # 除以255
            frame = frame.astype(np.float32) / 255.0
            return frame

    # 截图线程：持续捕获并缓存帧
    def capture_loop(self):
        last_capture = time.time()
        while not keyboard.is_pressed('esc'):
            # 控制截屏频率
            if time.time() - last_capture >= self.auto_interval:
                frame = self.capture_frame()
                # 维护缓存队列（若队列满则弹出旧帧）
                if self.frame_cache.full():
                    self.frame_cache.get()
                self.frame_cache.put(frame)
                last_capture = time.time()
            time.sleep(0.001)  # 降低CPU占用

    def process_npy(self, data):
        return np.array(data)

    # 推理线程：从缓存获取帧并推理
    def inference_loop(self):
        # 按esc结束
        while not keyboard.is_pressed('esc'):
            if self.frame_cache.qsize() == self.max_cache:
                frames = []
                for _ in range(self.max_cache):
                    frames.append(self.frame_cache.get())
                frames = np.array(frames)

                # 处理帧数据
                processed_frames = self.process_npy(frames)

                # 确保维度为3*240*320
                processed_frames = np.transpose(processed_frames, (0, 1, 2))

                # 转换为 (3, 1, H, W) 的 tensor
                image_tensor = torch.from_numpy(processed_frames).unsqueeze(1).float()

                # 预处理
                input_tensor = self.transform(image_tensor)

                # 添加批次维度，形状变为 [1, 3, 1, 224, 224] (B, T, C, H, W)
                input_tensor = input_tensor.unsqueeze(0)

                # 最终确保输入张量维度为 [B, 3, 1, 224, 224]
                input_tensor = input_tensor.to(self.device)

                # 模型推理
                with torch.no_grad():
                    output = self.model(input_tensor)
                    action = torch.argmax(output, dim=1).item()

                # 执行键盘操作
                print(f"模型输出: {action}")
                self.controller.perform_action(action)
            time.sleep(0.001)

# ====================== 主程序入口 ======================
if __name__ == "__main__":
    # 参数配置
    MODEL_PATH = "weights/CurrentBestModel.pth"
    CAPTURE_REGION = {"top": 0, "left": 0, "width": 800, "height": 600}  # 自定义截屏区域

    # 初始化并启动
    rt_inference = RealTimeInference(
        model_path=MODEL_PATH,
        region=CAPTURE_REGION,
        target_size=(224, 224)
    )

    # 启动双线程
    capture_thread = Thread(target=rt_inference.capture_loop, daemon=True)
    inference_thread = Thread(target=rt_inference.inference_loop, daemon=True)
    capture_thread.start()
    inference_thread.start()

    # 主线程等待退出
    print("系统已启动，按ESC键退出...")
    while not keyboard.is_pressed('esc'):
        time.sleep(1)
    print("系统退出")