from typing import Optional

import torch.nn as nn
import torch
import math
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """多头注意力机制实现"""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0  # 确保d_model能被n_heads整除

        self.d_model = d_model  # 模型维度
        self.n_heads = n_heads  # 注意力头数
        self.d_k = d_model // n_heads  # 每个头的维度

        # 线性变换层：Q, K, V
        self.w_q = nn.Linear(d_model, d_model, bias=False)  # Query权重矩阵
        self.w_k = nn.Linear(d_model, d_model, bias=False)  # Key权重矩阵
        self.w_v = nn.Linear(d_model, d_model, bias=False)  # Value权重矩阵
        self.w_o = nn.Linear(d_model, d_model, bias=False)  # 输出权重矩阵

        self.dropout = nn.Dropout(dropout)  # Dropout层
        self.scale = math.sqrt(self.d_k)  # 缩放因子，防止softmax饱和

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        Args:
            query: [batch_size, seq_len, d_model]
            key: [batch_size, seq_len, d_model]
            value: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len, seq_len] 或 None
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = query.shape

        # 分别获取序列长度
        tgt_seq_len = query.size(1)  # 目标序列长度
        src_seq_len = key.size(1)  # 源序列长度

        # 线性变换得到Q, K, V: [batch_size, seq_len, d_model]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        # 重塑为多头形式: [batch_size, n_heads, seq_len, d_k]
        Q = Q.view(batch_size, tgt_seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, src_seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, src_seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # 计算注意力
        attention_output = self.scaled_dot_product_attention(Q, K, V, mask)  # [batch_size, n_heads, seq_len, d_k]

        # 拼接多头结果: [batch_size, seq_len, d_model]
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )

        # 最终线性变换: [batch_size, seq_len, d_model]
        output = self.w_o(attention_output)

        return output

    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor,
                                     V: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        缩放点积注意力
        Args:
            Q: [batch_size, n_heads, seq_len, d_k]
            K: [batch_size, n_heads, seq_len, d_k]
            V: [batch_size, n_heads, seq_len, d_k]
            mask: [batch_size, seq_len, seq_len] 或 None
        Returns:
            output: [batch_size, n_heads, seq_len, d_k]
        """
        # 计算注意力分数: [batch_size, n_heads, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # 应用掩码（如果提供）
        if mask is not None:
            # 扩展mask维度以匹配scores的形状: [batch_size, 1, seq_len, seq_len]
            mask = mask.unsqueeze(1)
            scores.masked_fill_(mask == 0, -1e9)  # 将掩码位置设为很大的负值

        # 应用softmax: [batch_size, n_heads, seq_len, seq_len]
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)  # 应用dropout

        # 计算加权值: [batch_size, n_heads, seq_len, d_k]
        output = torch.matmul(attention_weights, V)

        return output