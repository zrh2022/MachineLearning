import torch.nn as nn
import torch
import math
import torch.nn.functional as F


class PositionwiseFeedForward(nn.Module):
    """位置前馈神经网络"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)  # 第一个线性层
        self.linear2 = nn.Linear(d_ff, d_model)  # 第二个线性层
        self.dropout = nn.Dropout(dropout)  # Dropout层

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        # FFN(x) = max(0, xW1 + b1)W2 + b2
        # 第一层 + ReLU + Dropout: [batch_size, seq_len, d_ff]
        hidden = self.dropout(F.relu(self.linear1(x)))
        # 第二层: [batch_size, seq_len, d_model]
        output = self.linear2(hidden)
        return output