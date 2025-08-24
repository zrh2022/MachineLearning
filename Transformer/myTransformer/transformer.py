import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from multi_head_attention import MultiHeadAttention
from layer_norm import LayerNorm
from feed_layer import PositionwiseFeedForward
from seq_pre_handler import TextPreHandler
from typing import Optional, Tuple
from transformer_decoder import TransformerDecoder
from transformer_encoder import TransformerEncoder


class Transformer(nn.Module):
    """完整的Transformer模型"""

    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, d_model: int = 512,
                 n_heads: int = 8, n_layers: int = 6, d_ff: int = 2048,
                 max_len: int = 5000, dropout: float = 0.1):
        super(Transformer, self).__init__()

        # 编码器和解码器
        self.encoder = TransformerEncoder(src_vocab_size, d_model, n_heads,
                                          n_layers, d_ff, max_len, dropout)
        self.decoder = TransformerDecoder(tgt_vocab_size, d_model, n_heads,
                                          n_layers, d_ff, max_len, dropout)

        # 输出投影层
        self.output_projection = nn.Linear(d_model, tgt_vocab_size, bias=False)  # [d_model] -> [tgt_vocab_size]

        # 权重共享：嵌入层和输出投影层共享权重（论文中的优化技巧）
        self.output_projection.weight = self.decoder.embedding.weight

        # 参数初始化
        self.init_parameters()

    def init_parameters(self):
        """参数初始化 - 使用Xavier均匀分布初始化"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        Args:
            src: [batch_size, src_seq_len] - 源序列
            tgt: [batch_size, tgt_seq_len] - 目标序列
            src_mask: [batch_size, src_seq_len, src_seq_len] - 源序列注意力掩码
            tgt_mask: [batch_size, tgt_seq_len, tgt_seq_len] - 目标序列注意力掩码
            src_key_padding_mask: [batch_size, src_seq_len] - 源序列填充掩码
            tgt_key_padding_mask: [batch_size, tgt_seq_len] - 目标序列填充掩码
        Returns:
            output: [batch_size, tgt_seq_len, tgt_vocab_size] - 输出logits
        """
        # 编码器前向传播: [batch_size, src_seq_len, d_model]
        encoder_output = self.encoder(src, src_mask)

        # 解码器前向传播: [batch_size, tgt_seq_len, d_model]
        decoder_output = self.decoder(tgt, encoder_output, tgt_mask, src_mask)

        # 输出投影: [batch_size, tgt_seq_len, tgt_vocab_size]
        output = self.output_projection(decoder_output)

        return output

    def generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """
        生成下三角掩码矩阵，用于防止解码器看到未来信息
        Args:
            sz: 序列长度
            device: 设备
        Returns:
            mask: [sz, sz] - 下三角掩码矩阵
        """
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)  # 上三角矩阵
        return mask == 0  # 转换为下三角掩码（True表示允许注意力）

    def create_padding_mask(self, seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
        """
        创建填充掩码
        Args:
            seq: [batch_size, seq_len] - 输入序列
            pad_idx: 填充token的索引
        Returns:
            mask: [batch_size, seq_len] - 填充掩码（True表示有效token）
        """
        return seq != pad_idx