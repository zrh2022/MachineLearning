import torch.nn as nn
import torch
import math


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()

        # 创建位置编码矩阵: [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()  # [max_len, 1]

        # 计算除数项
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))  # [d_model//2]

        # 应用sin到偶数索引
        pe[:, 0::2] = torch.sin(position * div_term)  # [max_len, d_model//2]
        # 应用cos到奇数索引
        pe[:, 1::2] = torch.cos(position * div_term)  # [max_len, d_model//2]

        # 添加batch维度: [1, max_len, d_model]
        pe = pe.unsqueeze(0)

        # 注册为buffer，不参与梯度计算但会保存到模型状态
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        # 添加位置编码: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :seq_len].detach()  # detach防止位置编码参与梯度计算, 拿到的是max_len，实际计算的是sql_len
        return x