import torch
import torch.nn as nn
import numpy as np
import keyboard
import time
from datetime import datetime
import os
from PIL import Image, ImageFilter
from sl_worker_newn import TemporalSpatialModel
import mss
import threading
from threading import Thread, Lock
from queue import Queue, Full
from collections import deque
import torch.nn.functional as F
from torch.distributions import Categorical
import pyautogui  # 鼠标操作
import pytesseract  # OCR文本识别
import cv2
import torchvision.transforms as transforms
import torch.nn.functional as F


# PPO代理
class PPOAgent:
    def __init__(self, action_dim=5, device='cuda', gae_lambda=0.93, clip_epsilon=0.2, clip_grad_norm=0.5):
        self.policy = TemporalSpatialModel(num_classes=action_dim).to(device)
        self.value_net = nn.Sequential(
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        ).to(device)
        self.optimizer = torch.optim.AdamW([
            {"params": self.policy.parameters()},
            {"params": self.value_net.parameters()}
        ], lr=0.0001, weight_decay=5e-5)
        self.clip_grad_norm = clip_grad_norm
        self.device = device
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.gamma = 0.98

    def get_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        if len(state.shape) == 4:
            state = state.unsqueeze(0)

        # 保存原始训练模式
        was_training = self.policy.training

        try:
            # 设置为评估模式
            self.policy.eval()

            with torch.no_grad():
                logits = self.policy(state)
                probs = torch.softmax(logits, dim=-1)

            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            return action.item(), log_prob.item() if state.shape[0] == 1 else (action, log_prob)
        finally:
            # 恢复原始训练模式
            self.policy.train(was_training)

    def update(self, states, actions, log_probs_old, rewards, dones, punish_indices=None):
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        log_probs_old = torch.FloatTensor(log_probs_old).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        with torch.no_grad():
            fused_feat = self.policy.extract_features(states)

        logits = self.policy.classifier(fused_feat)
        values = self.value_net(fused_feat).squeeze()

        next_values = torch.zeros_like(values)
        next_values[:-1] = values[1:]
        deltas = rewards + self.gamma * next_values * (1 - dones) - values

        advantages = torch.zeros_like(deltas)
        advantage = 0
        for t in reversed(range(len(deltas))):
            advantage = deltas[t] + self.gamma * self.gae_lambda * (1 - dones[t]) * advantage
            advantages[t] = advantage

        if punish_indices is not None:
            advantages[punish_indices] -= 2

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        ratio = torch.exp(log_probs - log_probs_old)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        target_values = rewards + self.gamma * next_values * (1 - dones)
        value_loss = F.mse_loss(values, target_values)

        total_loss = policy_loss + 0.5 * value_loss
        self.optimizer.zero_grad()
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.clip_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.clip_grad_norm)
        self.optimizer.step()

        return total_loss.item()


class GameEnvironment:
    def __init__(self, region=None):
        self.region = region
        self.screenshot_running = False
        self.screenshot_queue = Queue(maxsize=3)
        self.screenshot_lock = Lock()
        self.screenshot_thread = None
        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: F.interpolate(x, size=(224, 224),
                                                      mode='bilinear',
                                                      align_corners=False))
        ])

        # 文本检测相关配置
        self.text_check_running = False
        self.text_check_thread = None
        self.first_start = True
        self.game_over = False  # 游戏结束标志
        self.start_next_episode = False
        self.game_over_lock = Lock()
        # Play检测区域
        self.play_detect_region = (448, 484, 95, 40)  # (left, top, width, height)
        self.play_button_pos = (490,495)
        # Save检测区域
        self.save_detect_region = (330, 230, 88, 50)
        self.save_click_pos = (640, 263)
        # 模版匹配
        self.save_template = None  # 保存模板图像
        self.save_template_threshold = 0.7  # 匹配阈值
        self.play_template = None  # 保存Play模板图像
        self.play_template_threshold = 0.6  # Play匹配阈值
        self.load_save_template()  # 加载模板
        self.load_play_template()  # 加载Play模板
        self.save_image_count = 0

        self.n_key_pressed = False
        self.n_key_lock = Lock()
        self.n_reward_indices = []
        self.n_key_check_thread = None
        self.n_key_check_running = False
        self.action_mapping = {
            0: None,  # 上跳
            1: 'down',  # 下蹲/滑行
            2: 'left',  # 左切换车道
            3: 'right',  # 右切换车道
            4: 'up',  # 跳跃（高跳）
        }
        self.current_lane = 1
        self.last_reward = 0
        self.last_action = None
        self.reward_memory = deque(maxlen=10000)
        self.action_history = deque(maxlen=6)
        self.current_episode_step = 0  # 当前回合步数
        self.last_survival_time = 0  # 记录上一回合生存时间
        self.current_survival_start = 0  # 当前回合开始时间

        # 启动线程
        self.start_screenshot_thread()
        self.start_text_check_thread()
        self.start_n_key_check_thread()

    def load_save_template(self):
        """加载Save按钮模板图像"""
        try:
            template_path = r"C:\Users\4h55\Pictures\Camera Roll\c40499055919211021dbe7bd171fc5c.png"
            self.save_template = cv2.imread(template_path, 0)  # 灰度模式读取
            if self.save_template is None:
                raise FileNotFoundError(f"模板文件未找到：{template_path}")
        except Exception as e:
            print(f"加载Save模板失败：{str(e)}")
            self.save_template = None

    def load_play_template(self):
        """加载Play按钮模板图像"""
        try:
            # 模板图片路径需根据实际情况调整，假设与脚本同目录
            template_path = r"C:\Users\4h55\Pictures\Camera Roll\e73d94e580d4c7cc4d98cab51aa5c45.png"
            self.play_template = cv2.imread(template_path, 0)  # 灰度模式读取
            if self.play_template is None:
                raise FileNotFoundError(f"Play模板文件未找到：{template_path}")
        except Exception as e:
            print(f"加载Play模板失败：{str(e)}")
            self.play_template = None

    def start_text_check_thread(self):
        """启动文本检测线程"""
        if not self.text_check_thread or (self.text_check_thread and not self.text_check_thread.is_alive()):
            self.text_check_running = True
            self.text_check_thread = Thread(target=self.text_check_loop, daemon=True)
            self.text_check_thread.start()
            print("文本检测线程已启动")

    def text_check_loop(self):
        """分区域检测不同状态并触发操作"""
        while self.text_check_running:
            # 1. 优先检测Save标志（模板匹配方式）
            if self.save_template is not None:
                # 截取检测区域并转为灰度图
                save_region = self.save_detect_region
                with mss.mss() as sct:
                    monitor = {"top": save_region[1], "left": save_region[0], "width": save_region[2], "height": save_region[3]}
                    sct_img = sct.grab(monitor)
                    save_img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
                save_gray = save_img.convert('L')  # 转为灰度图
                save_np = np.array(save_gray)  # 转为numpy数组

                # 模板匹配
                result = cv2.matchTemplate(save_np, self.save_template, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

                # 判断是否超过阈值
                if max_val >= self.save_template_threshold:
                    with self.game_over_lock:
                        if not self.game_over:
                            self.game_over = True  # 标记游戏结束
                            pyautogui.click(*self.save_click_pos)  # 点击保存按钮
                            print("检测到Save标志")
                    time.sleep(1.0)  # 防重复触发
                    continue  # 检测到Save后跳过后续检测
            # 2. 仅在游戏结束状态下检测Play（触发下一轮启动）
            with self.game_over_lock:
                if self.game_over:  # 仅当游戏已结束时才检测启动信号
                    if self.play_template is not None:
                        # 截取检测区域并转为灰度图
                        play_region = self.play_detect_region
                        with mss.mss() as sct:
                            monitor = {"top": play_region[1], "left": play_region[0], "width": play_region[2],
                                       "height": play_region[3]}
                            sct_img = sct.grab(monitor)
                            play_img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
                        play_gray = play_img.convert('L')  # 转为灰度图
                        play_np = np.array(play_gray)  # 转为numpy数组

                        # 模板匹配
                        result = cv2.matchTemplate(play_np, self.play_template, cv2.TM_CCOEFF_NORMED)
                        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

                        # 判断是否超过阈值
                        if max_val >= self.play_template_threshold:
                            pyautogui.click(*self.play_button_pos)  # 点击启动位置
                            print("检测到Play")
                            # 再次确认点击是否成功
                            time.sleep(0.5)  # 等待界面响应
                            play_region = self.play_detect_region
                            with mss.mss() as sct:
                                monitor = {"top": play_region[1], "left": play_region[0], "width": play_region[2],
                                           "height": play_region[3]}
                                sct_img = sct.grab(monitor)
                                play_img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
                            play_gray = play_img.convert('L')  # 转为灰度图
                            play_np = np.array(play_gray)  # 转为numpy数组

                            # 再次进行模板匹配
                            result = cv2.matchTemplate(play_np, self.play_template, cv2.TM_CCOEFF_NORMED)
                            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

                            if max_val < self.play_template_threshold:
                                self.start_next_episode = True  # 标记需要启动下一轮
                                print("成功点击Play，开始下一轮")
                                time.sleep(0.5)  # 等待界面响应
                                continue  # 触发点击后跳过后续检测
                        time.sleep(0.5)
                        pyautogui.click(*self.save_click_pos)


            time.sleep(0.2)  # 检测间隔

    def start_n_key_check_thread(self):
        if not self.n_key_check_thread or (self.n_key_check_thread and not self.n_key_check_thread.is_alive()):
            self.n_key_check_running = True
            self.n_key_check_thread = Thread(target=self.n_key_check_loop, daemon=True)
            self.n_key_check_thread.start()

    def n_key_check_loop(self):
        while self.n_key_check_running:
            if keyboard.is_pressed('e'):
                with self.n_key_lock:
                    if not self.n_key_pressed:
                        self.n_key_pressed = True
                time.sleep(0.3)
            else:
                with self.n_key_lock:
                    self.n_key_pressed = False
            time.sleep(0.01)

    def start_screenshot_thread(self):
        thread_exists = hasattr(self, 'screenshot_thread') and self.screenshot_thread is not None
        thread_is_alive = thread_exists and self.screenshot_thread.is_alive()
        if not thread_is_alive:
            print("Starting screenshot thread...")
            self.screenshot_running = True
            self.screenshot_thread = threading.Thread(target=self.screenshot_loop)
            self.screenshot_thread.daemon = True
            self.screenshot_thread.start()
            print("Screenshot thread started successfully.")
        else:
            print("Screenshot thread is already running.")

    def stop_screenshot_thread(self):
        if hasattr(self, 'screenshot_thread') and self.screenshot_thread is not None:
            print("Stopping screenshot thread...")
            self.screenshot_running = False
            self.screenshot_thread.join(timeout=1.0)
            if self.screenshot_thread.is_alive():
                print("Warning: Screenshot thread is still alive after join.")
            else:
                print("Screenshot thread stopped successfully.")
        else:
            print("Screenshot thread is not running.")

    def screenshot_loop(self):
        print("Screenshot loop started.")
        try:
            while self.screenshot_running:
                start_time = time.time()
                frame = self.capture_frame()
                processed = self._preprocess(frame)
                with self.screenshot_lock:
                    if self.screenshot_queue.full():
                        self.screenshot_queue.get()
                    self.screenshot_queue.put(processed)
                elapsed = time.time() - start_time
                sleep_time = max(0.05 - elapsed, 0)
                time.sleep(sleep_time)
        except Exception as e:
            print(f"Error in screenshot loop: {e}")
        finally:
            print("Screenshot loop exited.")
            self.screenshot_running = False

    def capture_frame(self):
        try:
            with mss.mss() as sct:
                # 截图区域为 (0, 0) 到 (800, 600)
                monitor = self.region or {"top": 0, "left": 0, "width": 800, "height": 600}
                img = sct.grab(monitor)
                # 转换为灰度图
                frame = np.array(img, dtype=np.uint8)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
                # 用lanczo下采样到320*240
                frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_LANCZOS4)
                # 用非锐化掩蔽增强版进行锐化
                frame = self.unsharp_mask(frame)
                # 归一化
                frame = frame.astype(np.float32) / 255.0
                return frame
        except Exception as e:
            print(f"Error capturing frame: {e}")
            return None

    def _preprocess(self, frame):
        if frame is None:
            return np.zeros((224, 224), dtype=np.float32)

        # 将 numpy 数组转换为 torch.Tensor
        frame_tensor = torch.from_numpy(frame).float()

        # 确保添加必要的维度 (H,W) -> (C,H,W) -> (B,C,H,W)
        if frame_tensor.dim() == 2:  # (H,W)
            frame_tensor = frame_tensor.unsqueeze(0)  # (1,H,W)
        if frame_tensor.dim() == 3:  # (C,H,W)
            frame_tensor = frame_tensor.unsqueeze(0)  # (1,C,H,W)

        # 下采样
        frame_downsampled = self.transform(frame_tensor)

        # 转换为numpy数组 (H,W)
        return frame_downsampled.squeeze(0).squeeze(0).numpy()  # 移除批次和通道维度

    def unsharp_mask(self, image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
        blurred = cv2.GaussianBlur(image, kernel_size, sigma)
        sharpened = float(amount + 1) * image - float(amount) * blurred
        sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
        sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
        sharpened = sharpened.round().astype(np.uint8)
        if threshold > 0:
            low_contrast_mask = np.absolute(image - blurred) < threshold
            np.copyto(sharpened, image, where=low_contrast_mask)
        return sharpened

    def _get_state(self):
        frames = []
        wait_time = 0
        max_wait = 0.5  # 最大等待时间0.5秒

        # 非阻塞方式等待队列填充
        while wait_time < max_wait and len(frames) < 3:
            with self.screenshot_lock:
                available = self.screenshot_queue.qsize()
                if available > 0:
                    # 获取队列中所有可用帧（最多3帧）
                    frames = list(self.screenshot_queue.queue)[-min(available, 3):]

            if len(frames) < 3:
                time.sleep(0.05)
                wait_time += 0.05

        # 缺失帧时报错
        if len(frames) < 3:
            print("帧数不足，训练中止")
            time.sleep(5)

        state = np.stack(frames, axis=0)
        return np.expand_dims(state, axis=1)  # 添加通道维度

    def step(self, action):
        self._execute_action(action)
        state = self._get_state()
        reward = self._calculate_reward(state, action)
        with self.n_key_lock:
            if self.n_key_pressed:
                current_step = self.current_episode_step
                target_indices = [current_step - 2, current_step - 3, current_step - 4]
                valid_indices = [idx for idx in target_indices if idx >= 0]
                self.n_reward_indices.extend(valid_indices)
                self.n_key_pressed = False
        with self.game_over_lock:
            done = self.game_over
        self.reward_memory.append(reward)
        self.last_reward = reward
        self.last_action = action
        self.action_history.append(action)

        survival_time = time.time() - self.current_survival_start
        time_bonus = 0.05 * survival_time
        reward += time_bonus

        if done:
            with self.game_over_lock:
                if self.action_history:
                    self.action_history.pop()
                    self.action_history.pop()
        self.current_episode_step += 1

        return state, reward, done, {}

    def _execute_action(self, action):
        if action in self.action_mapping:
            key = self.action_mapping[action]

            # # 添加物理锁机制
            # execute_action = True
            # if action == 2 and self.current_lane == 0:
            #     #print("物理锁：已在最左车道，阻止向左移动")
            #     execute_action = False
            # elif action == 3 and self.current_lane == 2:
            #     #print("物理锁：已在最右车道，阻止向右移动")
            #     execute_action = False

            # if execute_action:
            print(f"action: {key}")

            if key:
                keyboard.press(key)
                time.sleep(0.03)
                keyboard.release(key)

                # # 更新车道位置（仅在成功执行动作时）
                # if key == 'left' and self.current_lane > 0:
                #     self.current_lane -= 1
                # elif key == 'right' and self.current_lane < 2:
                #     self.current_lane += 1

    def _calculate_reward(self, state, action):
        # 基础奖励
        reward = 0.2
        # # 边界限制
        # if (action == 2 and self.current_lane == 0) or (action == 3 and self.current_lane == 2):
        #     reward -= 1
        # 动作多样性奖惩
        if not self.last_action and action:
            reward -= 0.01
        if (len(self.action_history) >= 4) and action:
            repeat = 1
            action_h = action
            while action_h == self.action_history[-repeat] and repeat <= len(self.action_history) - 1:
                reward -= 0.03 * repeat
                action_h = self.action_history[-repeat]
                repeat += 1
        # 历史平滑
        if len(self.reward_memory) > 0:
            avg_reward = sum(self.reward_memory) / len(self.reward_memory)
            reward = 0.9 * reward + 0.1 * avg_reward
        return reward

    def reset(self):
        print("Resetting game environment...")
        with self.game_over_lock:
            self.game_over = False
        self.start_next_episode = False

        # 清空现有队列
        with self.screenshot_lock:
            while not self.screenshot_queue.empty():
                self.screenshot_queue.get()

        # 首次启动特殊处理
        if self.first_start:
            print("按下Q开始第一轮游戏...")
            # 等待时允许截图线程填充队列
            while not keyboard.is_pressed('q'):
                time.sleep(0.05)
            print("检测到Q键按下，开始第一轮游戏！")
            self.first_start = False
            # Q按下后额外等待0.3秒确保游戏开始
            time.sleep(0.3)
        else:
            time.sleep(1.0)  # 非首次启动等待1秒

        # 获取初始状态
        state = self._get_state()

        # 重置环境变量
        self.current_lane = 1
        self.last_reward = 0
        self.last_action = None
        self.reward_memory.clear()
        self.action_history.clear()
        self.current_episode_step = 0
        self.last_survival_time = 0
        self.current_survival_start = time.time()
        return state

    def close(self):
        # 原逻辑保持不变
        self.stop_screenshot_thread()
        self.stop_text_check_thread()
        self.stop_n_key_check_thread()
        print("游戏环境已关闭")

# 主训练循环
def main_rl_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    MODEL_PATH = r"E:\pythonProject\PythonProject4\1_2d_newmodel.pth"
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    env = GameEnvironment(region=None)
    action_dim = 5
    agent = PPOAgent(action_dim=action_dim, device=device)

    # 模型加载
    if os.path.exists(MODEL_PATH):
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=device)
            if 'policy_state_dict' not in checkpoint:
                agent.policy.load_state_dict(checkpoint)
                agent.policy.eval()
                print("加载监督学习预训练策略网络")
                # 冻结卷积层权重
                for name, param in agent.policy.named_parameters():
                    if 'fusion_block' in name or 'classifier' in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

                # 更新优化器参数
                optimizer_params = []
                for name, param in agent.policy.named_parameters():
                    if param.requires_grad:
                        optimizer_params.append(param)
                for param in agent.value_net.parameters():
                    optimizer_params.append(param)
                agent.optimizer = torch.optim.AdamW(optimizer_params, lr=1e-5, weight_decay=5e-5)
            else:
                agent.policy.load_state_dict(checkpoint['policy_state_dict'])
                agent.value_net.load_state_dict(checkpoint['value_net_state_dict'])
                agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                # 冻结卷积层权重
                for name, param in agent.policy.named_parameters():
                    if 'fusion_block' in name or 'classifier' in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

                # 更新优化器参数
                optimizer_params = []
                for name, param in agent.policy.named_parameters():
                    if param.requires_grad:
                        optimizer_params.append(param)
                for param in agent.value_net.parameters():
                    optimizer_params.append(param)
                agent.optimizer = torch.optim.AdamW(optimizer_params, lr=1e-5, weight_decay=5e-5)
                print("加载强化学习历史模型")
        except Exception as e:
            print(f"模型加载异常: {str(e)}，从头训练")
    else:
        print("未找到历史模型，从头训练")

    max_episodes = 1000000
    max_steps_per_episode = 1000000
    action_interval = 0.4

    # 初始状态获取
    state = env.reset()

    for episode in range(max_episodes):
        print(f"第 {episode + 1} 轮开始")
        episode_reward = 0
        done = False
        states, actions, log_probs, rewards, dones = [], [], [], [], []
        last_action_time = time.time()

        for step in range(max_steps_per_episode):
            current_time = time.time()
            time_to_wait = max(action_interval - (current_time - last_action_time), 0)
            time.sleep(time_to_wait)
            last_action_time = time.time()

            latest_state = env._get_state()
            action, log_prob = agent.get_action(latest_state)
            next_state, reward, done, _ = env.step(action)

            states.append(latest_state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(done)
            state = next_state
            episode_reward += reward

            if done:  # 检测到游戏结束
                print(f"第 {episode} 轮结束，步数 {step + 1}，奖励: {episode_reward:.2f}")
                # 奖励调整逻辑
                total_steps = len(rewards)
                start_idx = max(0, total_steps - 5)
                punish_indices = list(range(start_idx, total_steps))
                with env.n_key_lock:
                    reward_indices = env.n_reward_indices.copy()
                    env.n_reward_indices = []
                for idx in reward_indices:
                    if 0 <= idx < len(rewards):
                        rewards[idx] += 0.2
                for idx in punish_indices:
                    if 0 <= idx < len(rewards):
                        rewards[idx] -= 2.0 if actions[idx] is not None else 1.0

                # 模型更新
                if len(states) > 0:
                    states_np = np.stack(states, axis=0)
                    loss = agent.update(states_np, actions, log_probs, rewards, dones, punish_indices)
                    print(f"策略更新完成，损失: {loss:.4f}")

                # 保存模型
                torch.save({
                    'policy_state_dict': agent.policy.state_dict(),
                    'value_net_state_dict': agent.value_net.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                    'train_info': {
                        'episode': episode,
                        'last_reward': episode_reward,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                }, MODEL_PATH)
                print(f"模型已保存至 {MODEL_PATH}")

                # 等待下一轮启动信号
                print(f"等待第 {episode + 1} 轮启动信号...")
                while True:
                    with env.game_over_lock:
                        game_over = env.game_over
                    start_next = env.start_next_episode
                    if game_over and start_next:
                        break
                    time.sleep(0.5)
                state = env.reset()  # 重置环境并开始下一轮
                break

    env.close()


if __name__ == "__main__":
    main_rl_training()