import torch
import torch.nn as nn
import numpy as np


class DqnNet(nn.Module):
    def __init__(self):
        super(DqnNet, self).__init__()
        self.epsilon = 0.1  # 探索概率
        self.action_size = 4

        self.fc1 = nn.Linear(12, 100)  # 输入层到隐藏层
        self.fc2 = nn.Linear(100, 100)  # 隐藏层到隐藏层
        self.fc3 = nn.Linear(100, 4)  # 隐藏层到输出层
        self.relu = nn.ReLU()  # 激活函数

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32).to("cuda")
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  #
        return x

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            qs = self.forward(state)
            return qs.data.argmax()

