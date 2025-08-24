from typing import Optional

import torch.nn as nn
import torch
import math
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """层归一化"""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))  # 缩放参数
        self.beta = nn.Parameter(torch.zeros(d_model))  # 偏移参数
        self.eps = eps  # 防止除零的小数值

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        # 计算均值和方差（在最后一个维度上）
        mean = x.mean(-1, keepdim=True)  # [batch_size, seq_len, 1]
        std = x.std(-1, keepdim=True)  # [batch_size, seq_len, 1]

        # 归一化: [batch_size, seq_len, d_model]
        normalized = (x - mean) / (std + self.eps)

        # 应用可学习参数: [batch_size, seq_len, d_model]
        output = self.gamma * normalized + self.beta

        return output
