import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from multi_head_attention import MultiHeadAttention
from layer_norm import LayerNorm
from feed_layer import PositionwiseFeedForward
from seq_pre_handler import TextPreHandler
from typing import Optional, Tuple

class EncoderLayer(nn.Module):
    """Transformer编码器层"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super(EncoderLayer, self).__init__()

        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)  # 自注意力
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)  # 前馈网络
        self.norm1 = LayerNorm(d_model)  # 第一个层归一化
        self.norm2 = LayerNorm(d_model)  # 第二个层归一化
        self.dropout = nn.Dropout(dropout)  # Dropout层

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        Args:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len, seq_len] 或 None
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        # 第一个子层：自注意力 + 残差连接 + 层归一化
        attn_output = self.self_attention(x, x, x, mask)  # [batch_size, seq_len, d_model]
        x = self.norm1(x + self.dropout(attn_output))  # [batch_size, seq_len, d_model]

        # 第二个子层：前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)  # [batch_size, seq_len, d_model]
        x = self.norm2(x + self.dropout(ff_output))  # [batch_size, seq_len, d_model]

        return x