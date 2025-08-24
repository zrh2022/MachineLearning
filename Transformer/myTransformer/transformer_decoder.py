import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple
from positional_encoding import PositionalEncoding
from decoder_layer import DecoderLayer


class TransformerDecoder(nn.Module):
    """Transformer解码器"""

    def __init__(self, vocab_size: int, d_model: int, n_heads: int,
                 n_layers: int, d_ff: int, max_len: int = 5000, dropout: float = 0.1):
        super(TransformerDecoder, self).__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)  # 词嵌入层
        self.positional_encoding = PositionalEncoding(d_model, max_len)  # 位置编码

        # 堆叠多个解码器层
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.dropout = nn.Dropout(dropout)  # Dropout层

    def forward(self, tgt: torch.Tensor, encoder_output: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        Args:
            tgt: [batch_size, tgt_seq_len] - 目标序列token ids
            encoder_output: [batch_size, src_seq_len, d_model] - 编码器输出
            tgt_mask: [batch_size, tgt_seq_len, tgt_seq_len] - 目标序列掩码
            src_mask: [batch_size, tgt_seq_len, src_seq_len] - 源序列掩码
        Returns:
            output: [batch_size, tgt_seq_len, d_model]
        """
        # 词嵌入 + 缩放 + 位置编码: [batch_size, tgt_seq_len, d_model]
        x = self.embedding(tgt) * math.sqrt(self.d_model)  # 论文中的缩放技巧
        x = self.positional_encoding(x)  # [batch_size, tgt_seq_len, d_model]
        x = self.dropout(x)  # [batch_size, tgt_seq_len, d_model]

        # 通过所有解码器层
        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask, src_mask)  # [batch_size, tgt_seq_len, d_model]

        return x