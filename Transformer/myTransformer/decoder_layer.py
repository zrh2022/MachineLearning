import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from multi_head_attention import MultiHeadAttention
from layer_norm import LayerNorm
from feed_layer import PositionwiseFeedForward
from seq_pre_handler import TextPreHandler
from typing import Optional, Tuple


class DecoderLayer(nn.Module):
    """Transformer解码器层"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super(DecoderLayer, self).__init__()

        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)  # 掩码自注意力
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)  # 编码器-解码器注意力
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)  # 前馈网络
        self.norm1 = LayerNorm(d_model)  # 第一个层归一化
        self.norm2 = LayerNorm(d_model)  # 第二个层归一化
        self.norm3 = LayerNorm(d_model)  # 第三个层归一化
        self.dropout = nn.Dropout(dropout)  # Dropout层

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
                self_attn_mask: Optional[torch.Tensor] = None,
                cross_attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        Args:
            x: [batch_size, tgt_seq_len, d_model] - 目标序列
            encoder_output: [batch_size, src_seq_len, d_model] - 编码器输出
            self_attn_mask: [batch_size, tgt_seq_len, tgt_seq_len] - 自注意力掩码
            cross_attn_mask: [batch_size, tgt_seq_len, src_seq_len] - 交叉注意力掩码
        Returns:
            output: [batch_size, tgt_seq_len, d_model]
        """
        # 第一个子层：掩码自注意力 + 残差连接 + 层归一化
        self_attn_output = self.self_attention(x, x, x, self_attn_mask)  # [batch_size, tgt_seq_len, d_model]
        x = self.norm1(x + self.dropout(self_attn_output))  # [batch_size, tgt_seq_len, d_model]

        # 第二个子层：编码器-解码器注意力 + 残差连接 + 层归一化
        cross_attn_output = self.cross_attention(x, encoder_output, encoder_output,
                                                 cross_attn_mask)  # [batch_size, tgt_seq_len, d_model]
        x = self.norm2(x + self.dropout(cross_attn_output))  # [batch_size, tgt_seq_len, d_model]

        # 第三个子层：前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)  # [batch_size, tgt_seq_len, d_model]
        x = self.norm3(x + self.dropout(ff_output))  # [batch_size, tgt_seq_len, d_model]

        return x