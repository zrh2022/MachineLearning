import numpy as np

from Bandit import Bandit
from Agent import Agent
import matplotlib.pyplot as plt

# k臂老虎机的臂数
arms = 10
# 随机"探索"的概率
epsilon = 0.2
# 行动次数
action_times = 1000
# 运行次数
runs = 200
all_win_rates = np.zeros((runs, action_times))

for run in range(runs):
    # k臂老虎机
    band = Bandit(arms)
    # 智能代理
    agent = Agent(epsilon, arms)
    total_reward = 0
    total_rewards = []
    win_rates = []
    total_actions = []

    for i in range(action_times):
        # 代理每次选择一个行动
        action = agent.get_action()
        # 玩索引为action的老虎机，获得奖励
        reward = band.play(action)
        # 智能代理进行更新引为action的老虎机的价值
        agent.update(action, reward)

        total_actions.append(action)
        # 更新当前总奖励
        total_reward += reward
        total_rewards.append(total_reward)
        win_rates.append(total_reward / (i + 1))

    all_win_rates[run] = win_rates

arg_rates = np.average(all_win_rates, axis=0)

# 设置字体为支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

plt.xlabel("次数")
plt.ylabel("平均胜率")
plt.plot(arg_rates)
plt.show()

# plt.xlabel("次数")
# plt.ylabel("胜率")
# for i in range(runs):
#     plt.plot(all_win_rates[i])
# plt.show()
