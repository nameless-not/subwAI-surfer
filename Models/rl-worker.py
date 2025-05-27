import torch
import torch.nn as nn
import numpy as np
import keyboard
import time
from datetime import datetime
import os
from PIL import Image, ImageFilter
from TemporalSpatialModel import TemporalSpatialModel  # 从模型文件导入
import pyautogui
from threading import Thread, Lock
from queue import Queue, Full


# 经验回放缓冲区类
class ReplayBuffer:
    def __init__(self, max_size=50000):
        self.max_size = max_size
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.lock = Lock()

    def add(self, state, action, log_prob, reward, done):
        with self.lock:
            self.states.append(state)
            self.actions.append(action)
            self.log_probs.append(log_prob)
            self.rewards.append(reward)
            self.dones.append(done)

            if len(self.states) > self.max_size:
                self.states.pop(0)
                self.actions.pop(0)
                self.log_probs.pop(0)
                self.rewards.pop(0)
                self.dones.pop(0)

    def get_all(self):
        with self.lock:
            return (
                np.array(self.states),
                np.array(self.actions),
                np.array(self.log_probs),
                np.array(self.rewards),
                np.array(self.dones)
            )

    def clear(self):
        with self.lock:
            self.states = []
            self.actions = []
            self.log_probs = []
            self.rewards = []
            self.dones = []


# -------------------------- 强化学习环境封装 --------------------------
class GameEnvironment:
    def __init__(self, region=None, data_path="rl_data/"):
        self.region = region  # 截图区域（格式：(left, top, width, height)）
        self.screenshot_queue = Queue(maxsize=3)  # 固定大小队列，最多缓存3帧
        self.screenshot_lock = Lock()  # 队列操作锁（线程安全）
        self.screenshot_thread = None  # 截图线程
        self.screenshot_running = False  # 线程运行标志
        self.start_screenshot_thread()  # 启动截图线程

        # 其他初始化
        self.last_action = None
        self.reset()

        # 奖励相关状态
        self.game_started = False
        self.start_time = None
        self.current_second = 0
        self.reward_events = {}
        self.end_triggered = False

    def start_screenshot_thread(self):
        """启动独立截图线程（0.05秒间隔）"""
        if self.screenshot_thread and self.screenshot_thread.is_alive():
            return

        self.screenshot_running = True

        def screenshot_loop():
            while self.screenshot_running:
                start_time = time.time()

                # 1. 截图并预处理
                frame = self._capture_frame()
                processed = self._preprocess(frame)

                # 2. 线程安全更新队列（保留最新3帧）
                with self.screenshot_lock:
                    try:
                        self.screenshot_queue.put(processed, block=False)
                    except Full:
                        self.screenshot_queue.get()  # 移除最旧帧
                        self.screenshot_queue.put(processed)  # 插入新帧

                # 3. 控制0.05秒间隔（补偿截图耗时）
                elapsed = time.time() - start_time
                sleep_time = max(0.05 - elapsed, 0)
                time.sleep(sleep_time)

        self.screenshot_thread = Thread(target=screenshot_loop, daemon=True)
        self.screenshot_thread.start()

    def stop_screenshot_thread(self):
        """停止截图线程"""
        self.screenshot_running = False
        if self.screenshot_thread:
            self.screenshot_thread.join()

    def _capture_frame(self):
        """使用pyautogui截图（支持指定区域）"""
        if self.region:
            screenshot = pyautogui.screenshot(region=self.region)
        else:
            screenshot = pyautogui.screenshot()
        return np.array(screenshot.convert('L'))  # 转换为灰度图（H, W）

    def _preprocess(self, frame):
        """预处理：缩放→锐化→归一化"""
        img = Image.fromarray(frame.astype(np.uint8))
        img = img.resize((224, 224), resample=Image.LANCZOS)  # 直接缩放到224x224
        img = img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=150, threshold=3))
        processed = np.array(img) / 255.0  # 归一化到0-1
        return processed

    def _get_state(self):
        """从队列获取最新3帧（保证时序性）"""
        with self.screenshot_lock:
            # 等待队列填满3帧（初始阶段）
            while self.screenshot_queue.qsize() < 3:
                time.sleep(0.001)

            # 取出所有帧（最多3帧）
            frames = []
            while not self.screenshot_queue.empty():
                frames.append(self.screenshot_queue.get())

            # 确保保留最新3帧（处理队列延迟）
            if len(frames) > 3:
                frames = frames[-3:]
            elif len(frames) < 3:
                frames = frames + [frames[-1]] * (3 - len(frames))  # 用最近帧填充

            # 重新放入未使用的帧（保证队列始终有最新数据）
            for frame in frames:
                try:
                    self.screenshot_queue.put(frame, block=False)
                except Full:
                    self.screenshot_queue.get()
                    self.screenshot_queue.put(frame)

            return np.stack(frames, axis=0)  # 形状(3, 224, 224)

    def reset(self):
        """重置环境时清空截图队列"""
        with self.screenshot_lock:
            while not self.screenshot_queue.empty():
                self.screenshot_queue.get()
        self.last_action = None
        self.game_started = False
        self.start_time = None
        self.current_second = 0
        self.reward_events = {}
        self.end_triggered = False
        return self._get_state()

    # 奖励计算、动作执行等逻辑
    def _update_reward_events(self):
        current_time = time.time()
        if not self.game_started:
            return
        elapsed = current_time - self.start_time
        self.current_second = int(elapsed)
        if keyboard.is_pressed('k'):
            self.reward_events.setdefault(self.current_second, []).append(3)
        if keyboard.is_pressed('m'):
            self.reward_events.setdefault(self.current_second, []).append(-3)

    def step(self, action):
        action_key = ['', 'w', 'a', 's', 'd'][action]
        if action_key:
            self._press_key(action_key)
        self._update_reward_events()
        new_state = self._get_state()
        reward = self._calculate_reward()
        done = self._check_termination()
        return new_state, reward, done, {}

    def _calculate_reward(self):
        base_reward = 0
        if self.current_second > 0:
            base_reward += 1
        event_reward = sum(self.reward_events.get(self.current_second, []))
        if self.end_triggered:
            elapsed_in_second = time.time() - self.start_time - self.current_second
            penalty = -10 * elapsed_in_second
            return base_reward + event_reward + penalty
        return base_reward + event_reward

    def _check_termination(self):
        if keyboard.is_pressed('space') and self.game_started:
            self.end_triggered = True
            time.sleep(1)
            return True
        return False

    def _press_key(self, key):
        keyboard.press(key)
        time.sleep(0.03)
        keyboard.release(key)


# -------------------------- 强化学习智能体 --------------------------
class PPOAgent:
    def __init__(self, state_dim, action_dim=5, device='cuda'):
        self.policy = TemporalSpatialModel(num_classes=action_dim).to(device)
        self.value_net = nn.Sequential(
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 1)
        ).to(device)
        self.optimizer = torch.optim.AdamW(
            list(self.policy.parameters()) + list(self.value_net.parameters()),
            lr=3e-4,
            weight_decay=3e-4
        )
        self.device = device
        self.clip_epsilon = 0.2
        self.gamma = 0.99

    def get_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        state_tensor = state_tensor.unsqueeze(2)
        logits = self.policy(state_tensor)
        probs = torch.softmax(logits, dim=-1)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob.item()

    def update(self, states, actions, log_probs, rewards, dones):
        states = torch.tensor(states, dtype=torch.float32, device=self.device).unsqueeze(2)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor(log_probs, dtype=torch.float32, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            values = self.value_net(self.policy(states).squeeze())
            advantages = self._compute_advantages(rewards, values, dones)

        new_logits = self.policy(states)
        new_probs = torch.softmax(new_logits, dim=-1)
        new_dist = torch.distributions.Categorical(new_probs)
        new_log_probs = new_dist.log_prob(actions)

        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        returns = advantages + values
        value_loss = nn.MSELoss()(self.value_net(self.policy(states).squeeze()), returns)
        total_loss = policy_loss + 0.5 * value_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.item()

    def _compute_advantages(self, rewards, values, dones):
        advantages = torch.zeros_like(rewards)
        gae = 0
        values = torch.cat([values, torch.tensor([0.0], device=self.device)])
        for t in reversed(range(len(rewards))):
            mask = 1 - dones[t].float()
            delta = rewards[t] + self.gamma * values[t + 1] * mask - values[t]
            gae = delta + self.gamma * 0.95 * gae * mask
            advantages[t] = gae
        return (advantages - advantages.mean()) / (advantages.std() + 1e-8)


# -------------------------- 主训练循环 --------------------------
def main_rl_training():
    env = GameEnvironment()
    agent = PPOAgent(state_dim=(3, 224, 224), action_dim=5)
    replay_buffer = ReplayBuffer()

    print("=== 训练说明 ===")
    print("空格：开始/重新开始游戏")
    print("游戏中按空格：结束当前局（不终止训练）")
    print("ESC：结束所有训练并触发模型更新")
    print("=================")

    training = True
    while training:
        print("\n按空格开始游戏...")
        keyboard.wait('space')
        env.reset()
        env.game_started = True
        env.start_time = time.time()
        print("游戏开始！")

        states, actions, log_probs, rewards, dones = [], [], [], [], []
        total_reward = 0

        while True:
            if keyboard.is_pressed('esc'):
                print("检测到 ESC 键，结束训练...")
                for s, a, lp, r, d in zip(states, actions, log_probs, rewards, dones):
                    replay_buffer.add(s, a, lp, r, d)
                training = False
                break

            state = env._get_state()
            action, log_prob = agent.get_action(state)
            new_state, reward, done, _ = env.step(action)
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(done)
            total_reward += reward

            if done:
                print(f"游戏结束！本局总奖励：{total_reward:.2f}")
                for s, a, lp, r, d in zip(states, actions, log_probs, rewards, dones):
                    replay_buffer.add(s, a, lp, r, d)
                break

    if len(replay_buffer.states) > 0:
        states, actions, log_probs, rewards, dones = replay_buffer.get_all()
        loss = agent.update(states, actions, log_probs, rewards, dones)
        print(f"模型更新完成，损失：{loss:.4f}")
    else:
        print("无有效数据，跳过更新")

    env.stop_screenshot_thread()
    print("训练结束")


if __name__ == "__main__":
    main_rl_training()