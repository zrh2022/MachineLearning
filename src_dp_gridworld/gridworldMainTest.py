from gridworld import Gridworld
import numpy as np
from evalpolicy import EvalPolicy

env = Gridworld()
rows = 3
columns = 4

print(env.action_space)
print('=====')
print(env.getStates())

values = {}
# 使用正态分布随机初始化，[-1,1],方差为0
for state in env.getStates():
    values[state] = np.random.randn()

# 获取策略
policy_eval = EvalPolicy()

# 执行策略
# values = policy_eval.policy_eval(policy_eval.pi, values, env, gamma=0.9)
# env.render_env(rows, columns, values, policy_eval.getSpeActions(policy_eval.pi))
#
# pi_greedy = policy_eval.getGreedyPolicy(policy_eval.pi, values, env, gamma=0.9)
# env.render_env(rows, columns, values, policy_eval.getSpeActions(pi_greedy))
# print(pi_greedy)

policy_eval.policy_greedy(policy_eval.pi, values, env, gamma=0.9)