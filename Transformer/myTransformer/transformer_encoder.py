import torch.nn as nn
import torch
import math
import torch.nn.functional as F

from encoder_layer import EncoderLayer
from multi_head_attention import MultiHeadAttention
from layer_norm import LayerNorm
from feed_layer import PositionwiseFeedForward
from seq_pre_handler import TextPreHandler
from typing import Optional, Tuple
from positional_encoding import PositionalEncoding


class TransformerEncoder(nn.Module):
    """Transformer编码器"""

    def __init__(self, vocab_size: int, d_model: int, n_heads: int,
                 n_layers: int, d_ff: int, max_len: int = 5000, dropout: float = 0.1):
        super(TransformerEncoder, self).__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)  # 词嵌入层
        self.positional_encoding = PositionalEncoding(d_model, max_len)  # 位置编码

        # 堆叠多个编码器层
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.dropout = nn.Dropout(dropout)  # Dropout层

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        Args:
            src: [batch_size, src_seq_len] - 源序列token ids
            src_mask: [batch_size, src_seq_len, src_seq_len] - 源序列掩码
        Returns:
            output: [batch_size, src_seq_len, d_model]
        """
        # 词嵌入 + 缩放 + 位置编码: [batch_size, src_seq_len, d_model]
        x = self.embedding(src) * math.sqrt(self.d_model)  # 论文中的缩放技巧
        x = self.positional_encoding(x)  # [batch_size, src_seq_len, d_model]
        x = self.dropout(x)  # [batch_size, src_seq_len, d_model]

        # 通过所有编码器层
        for layer in self.layers:
            x = layer(x, src_mask)  # [batch_size, src_seq_len, d_model]

        return x
