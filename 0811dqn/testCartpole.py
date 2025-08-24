import gym                                # 引入 Gym 环境
import torch                              # 引入 PyTorch
import torch.nn as nn                      # 神经网络模块
import torch.optim as optim                # 优化器模块
import numpy as np                         # 数值计算
import random                              # 随机数生成
import matplotlib.pyplot as plt            # 用于绘制奖励曲线

# =======================
# 设备配置
# =======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 使用 GPU（cuda）或 CPU
print("Using device:", device)  # 打印当前设备

# =======================
# 状态归一化函数
# =======================
def normalize_state(state):  # 将状态缩放到 [-1,1] 左右
    return np.array([
        state[0] / 4.8,    # 小车位置归一化
        state[1] / 5.0,    # 小车速度归一化
        state[2] / 0.418,  # 杆子角度归一化
        state[3] / 5.0     # 杆子角速度归一化
    ], dtype=np.float32)

# =======================
# 权重初始化
# =======================
def init_weights(m):  # 初始化网络参数
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)  # Xavier 初始化权重
        nn.init.constant_(m.bias, 0)       # 偏置初始化为 0

# =======================
# Dueling Double DQN 网络
# =======================
class DuelingQNet(nn.Module):
    def __init__(self, state_size, action_size):
        super(DuelingQNet, self).__init__()
        # 公共特征提取层
        self.feature = nn.Sequential(
            nn.Linear(state_size, 128),  # 输入状态维度 -> 128
            nn.ReLU()                     # ReLU 激活
        )
        # value 流
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),  # 隐藏层 128 -> 128
            nn.ReLU(),
            nn.Linear(128, 1)     # 输出状态价值
        )
        # advantage 流
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),           # 隐藏层 128 -> 128
            nn.ReLU(),
            nn.Linear(128, action_size)    # 输出每个动作的优势
        )

    def forward(self, x):
        x = self.feature(x)                          # 提取特征
        value = self.value_stream(x)                 # 计算状态价值
        advantage = self.advantage_stream(x)         # 计算动作优势
        return value + (advantage - advantage.mean(dim=1, keepdim=True))  # 合并 value 与 advantage

# =======================
# 优先经验回放
# =======================
class PERBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=1e-4):
        self.capacity = capacity                    # 回放池容量
        self.alpha = alpha                          # 优先级指数
        self.beta = beta                            # 重要性采样权重指数
        self.beta_increment = beta_increment        # 每次采样 beta 增量
        self.pos = 0                                # 当前存储位置
        self.size = 0                               # 当前存储大小
        # 初始化状态、动作、奖励、下一状态、done mask
        self.states = np.zeros((capacity, 4), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, 4), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.priorities = np.zeros(capacity, dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        """将新经验存入回放池"""
        max_priority = self.priorities.max() if self.size > 0 else 1.0  # 最大优先级
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = done
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity               # 循环覆盖
        self.size = min(self.size + 1, self.capacity)          # 更新大小

    def sample(self, batch_size):
        """按优先级采样 batch"""
        probs = self.priorities[:self.size] ** self.alpha        # 优先级指数
        probs /= probs.sum()                                     # 归一化概率
        indices = np.random.choice(self.size, batch_size, p=probs)
        weights = (self.size * probs[indices]) ** (-self.beta)   # 重要性采样权重
        self.beta = min(1.0, self.beta + self.beta_increment)   # beta 增量
        weights /= weights.max()                                 # 归一化
        # 转为 torch tensor
        return (
            torch.tensor(self.states[indices], dtype=torch.float32, device=device),
            torch.tensor(self.actions[indices], dtype=torch.long, device=device),
            torch.tensor(self.rewards[indices], dtype=torch.float32, device=device),
            torch.tensor(self.next_states[indices], dtype=torch.float32, device=device),
            torch.tensor(self.dones[indices], dtype=torch.float32, device=device),
            torch.tensor(weights, dtype=torch.float32, device=device),
            indices
        )

    def update_priorities(self, indices, priorities):
        """更新经验优先级"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return self.size

# =======================
# 软更新
# =======================
def soft_update(local_model, target_model, tau):
    """局部网络参数软更新到目标网络"""
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

# =======================
# 环境与超参数
# =======================
env = gym.make("CartPole-v1")              # 创建环境
state_size = env.observation_space.shape[0]  # 状态维度
action_size = env.action_space.n             # 动作维度

lr = 5e-5                  # 学习率
gamma = 0.99               # 折扣因子
epsilon_start = 1.0        # 初始 epsilon
epsilon_final = 0.01       # 最小 epsilon
epsilon_decay_steps = 300  # epsilon 衰减步数
episodes = 500             # 最大回合数
batch_size = 64
buffer_capacity = 10000    # 回放池容量
tau = 0.005

# =======================
# 网络与优化器
# =======================
policy_net = DuelingQNet(state_size, action_size).to(device)
target_net = DuelingQNet(state_size, action_size).to(device)
policy_net.apply(init_weights)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(policy_net.parameters(), lr=lr)
memory = PERBuffer(buffer_capacity)

# =======================
# ε 线性衰减函数
# =======================
def get_epsilon(step):
    if step >= epsilon_decay_steps:
        return epsilon_final
    return epsilon_start - (epsilon_start - epsilon_final) * (step / epsilon_decay_steps)

# =======================
# 训练循环 + 自动停止
# =======================
global_step = 0
reward_list = []  # 存储每回合 reward
solved = False    # 标记是否满足自动停止条件

for ep in range(episodes):
    state, _ = env.reset()                   # 重置环境
    state = normalize_state(state)           # 归一化状态
    total_reward = 0.0                        # 累积 reward
    done = False

    while not done:
        epsilon = get_epsilon(global_step)   # 获取当前 epsilon
        global_step += 1

        # ε-贪婪策略选择动作
        if random.random() < epsilon:
            action = env.action_space.sample()  # 随机动作
        else:
            with torch.no_grad():
                state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                action = torch.argmax(policy_net(state_t), dim=1).item()  # 最大 Q 动作

        # 与环境交互
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = normalize_state(next_state)
        done = terminated or truncated
        total_reward += reward                   # 累积 reward

        # 奖励归一化（保留原版逻辑）
        reward = reward / 10.0

        # 存入回放池
        memory.push(state, action, reward, next_state, done)
        state = next_state

        # 网络训练
        if len(memory) >= batch_size:
            states, actions, rewards, next_states, dones, weights, indices = memory.sample(batch_size)
            with torch.no_grad():
                next_actions = policy_net(next_states).argmax(1)
                next_q_values = target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                target_q = rewards + gamma * next_q_values * (1 - dones)
            current_q = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

            td_errors = target_q - current_q
            td_errors = torch.clamp(td_errors, -10, 10)           # TD-error 裁剪
            loss = (weights * td_errors.pow(2)).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
            optimizer.step()

            new_priorities = td_errors.abs().detach().cpu().numpy() + 1e-6
            memory.update_priorities(indices, new_priorities)
            soft_update(policy_net, target_net, tau)

    reward_list.append(total_reward)   # 保存回合 reward

    # 打印日志
    if (ep + 1) % 10 == 0:
        print(f"Episode {ep+1} | Reward: {total_reward:.0f} | Epsilon: {epsilon:.3f}")

    # 自动停止条件：最近 20 回合平均 reward >= 495
    if len(reward_list) >= 20 and np.mean(reward_list[-20:]) >= 495:
        print(f"Solved after {ep+1} episodes! Average reward: {np.mean(reward_list[-20:]):.2f}")
        solved = True
        break

env.close()  # 关闭环境

# =======================
# 绘制奖励曲线
# =======================
plt.plot(reward_list)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Training Reward Curve")
plt.show()

# =======================
# 倒立摆实际测试（可视化）
# =======================
test_episodes = 5
env = gym.make("CartPole-v1", render_mode="human")  # 开启可视化
for ep in range(test_episodes):
    state, _ = env.reset()
    state = normalize_state(state)
    done = False
    total_reward = 0
    while not done:
        env.render()  # 渲染画面
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            action = torch.argmax(policy_net(state_t), dim=1).item()
        next_state, reward, terminated, truncated, _ = env.step(action)
        state = normalize_state(next_state)
        done = terminated or truncated
        total_reward += reward
    print(f"Test Episode {ep+1} Reward: {total_reward}")
env.close()
