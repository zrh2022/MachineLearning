import gym                                # 引入 Gym 环境
import torch                              # 引入 PyTorch
import torch.nn as nn                      # 神经网络模块
import torch.optim as optim                # 优化器模块
import numpy as np                         # 数值计算
import random                              # 随机数生成

# =======================
# 设备配置
# =======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU 或 CPU
print("Using device:", device)  # 打印设备

# =======================
# 状态归一化函数
# =======================
def normalize_state(state):  # 将状态缩放到 [-1,1] 左右
    return np.array([
        state[0] / 4.8,
        state[1] / 5.0,
        state[2] / 0.418,
        state[3] / 5.0
    ], dtype=np.float32)

# =======================
# 权重初始化
# =======================
def init_weights(m):  # 初始化网络参数
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)  # Xavier 初始化
        nn.init.constant_(m.bias, 0)       # 偏置初始化 0

# =======================
# Dueling Double DQN 网络
# =======================
class DuelingQNet(nn.Module):
    def __init__(self, state_size, action_size):
        super(DuelingQNet, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        x = self.feature(x)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

# =======================
# 优先经验回放
# =======================
class PERBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=1e-4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.pos = 0
        self.size = 0
        self.states = np.zeros((capacity, 4), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, 4), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.priorities = np.zeros(capacity, dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        max_priority = self.priorities.max() if self.size > 0 else 1.0
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = done
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        probs = self.priorities[:self.size] ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(self.size, batch_size, p=probs)
        weights = (self.size * probs[indices]) ** (-self.beta)
        self.beta = min(1.0, self.beta + self.beta_increment)
        weights /= weights.max()
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
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return self.size

# =======================
# 软更新
# =======================
def soft_update(local_model, target_model, tau):
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

# =======================
# 环境与超参数
# =======================
env = gym.make("CartPole-v1")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

lr = 5e-5                  # 学习率降低
gamma = 0.99               # 折扣因子
epsilon_start = 1.0        # 初始 epsilon
epsilon_final = 0.01       # 最小 epsilon
epsilon_decay_steps = 300  # 线性衰减步数
episodes = 500             # 回合数
batch_size = 64
buffer_capacity = 10000    # 回放池容量适中
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
# 训练循环
# =======================
global_step = 0
for ep in range(episodes):
    state, _ = env.reset()
    state = normalize_state(state)
    total_reward = 0.0

    while True:
        epsilon = get_epsilon(global_step)              # 获取当前 epsilon
        global_step += 1

        # ε-贪婪动作选择
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                action = torch.argmax(policy_net(state_t), dim=1).item()

        # 与环境交互
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = normalize_state(next_state)
        done = terminated or truncated
        total_reward += reward

        # 奖励归一化
        reward = reward / 10.0

        # 存入回放池
        memory.push(state, action, reward, next_state, done)
        state = next_state

        # 训练网络
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

        if done:
            break

    if (ep + 1) % 10 == 0:
        print(f"Episode {ep+1} | Reward: {total_reward:.0f} | Epsilon: {epsilon:.3f}")

env.close()
