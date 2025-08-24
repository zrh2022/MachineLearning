import time

import gym
import random

# 创建 CartPole 环境
env = gym.make("CartPole-v1", render_mode="human")

# 尝试 10 次随机动作
for episode in range(10):
    observation = env.reset()
    total_reward = 0

    while True:
        # 随机选择动作（0=左推，1=右推）
        action = env.action_space.sample()

        # 执行动作（适配新旧版本）
        observation, reward, done, *_ = env.step(action)
        total_reward += reward

        print(action, observation, reward, done, total_reward)
        # 渲染环境（可选）
        env.render()
        time.sleep(0.1)

        # 结束条件
        if done:
            print(f"Episode {episode + 1}: Total Reward = {total_reward}")
            break

# 关闭环境
env.close()