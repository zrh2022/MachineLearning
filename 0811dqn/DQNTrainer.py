import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class DQNTrainer:
    def __init__(self, model, learning_rate=0.001):
        self.gamma = 0.9  # 折现率
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    def update(self, state, action, reward, next_state, done):
        if done:
            next_q = torch.zeros(1).to("cuda")
        else:
            with torch.no_grad():
                next_qs = self.model.forward(next_state)
                next_q = next_qs.max(axis=1).values


        target = self.gamma * next_q + reward
        qs = self.model.forward(state)
        q = qs[:, action]
        loss = nn.MSELoss()(target, q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.data

    def train(self, X_train, y_train, epochs=100, batch_size=32):
        # 转换数据为PyTorch张量
        X_tensor = torch.FloatTensor(X_train)
        y_tensor = torch.FloatTensor(y_train)

        # 创建数据集和数据加载器
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # 训练循环
        for epoch in range(epochs):
            for batch_X, batch_y in dataloader:
                # 前向传播
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)

                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    def predict(self, X_test):
        # 转换数据为PyTorch张量
        X_tensor = torch.FloatTensor(X_test)

        # 设置为评估模式
        self.model.eval()

        # 不计算梯度
        with torch.no_grad():
            predictions = self.model(X_tensor)

        # 返回numpy数组
        return predictions.numpy()
