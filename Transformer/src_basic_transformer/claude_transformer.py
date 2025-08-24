import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple


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
        x = x + self.pe[:, :seq_len].detach()  # detach防止位置编码参与梯度计算
        return x


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


def test_transformer():
    """测试Transformer模型"""
    print("开始测试Transformer模型...")

    # 检查GPU是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 模型参数
    src_vocab_size = 1000  # 源语言词汇表大小
    tgt_vocab_size = 1000  # 目标语言词汇表大小
    d_model = 512  # 模型维度
    n_heads = 8  # 注意力头数
    n_layers = 6  # 层数
    d_ff = 2048  # 前馈网络隐藏层维度
    max_len = 100  # 最大序列长度
    dropout = 0.1  # Dropout概率

    # 创建模型并移动到GPU
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_len=max_len,
        dropout=dropout
    ).to(device)

    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

    # 创建测试数据
    batch_size = 32
    src_seq_len = 20
    tgt_seq_len = 25

    # 随机生成源序列和目标序列 (避免使用pad_idx=0)
    src = torch.randint(1, src_vocab_size, (batch_size, src_seq_len)).to(device)  # [batch_size, src_seq_len]
    tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_seq_len)).to(device)  # [batch_size, tgt_seq_len]

    print(f"源序列形状: {src.shape}")
    print(f"目标序列形状: {tgt.shape}")

    # 创建掩码
    # 目标序列的因果掩码（防止看到未来信息）
    tgt_mask = model.generate_square_subsequent_mask(tgt_seq_len, device)  # [tgt_seq_len, tgt_seq_len]
    tgt_mask = tgt_mask.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, tgt_seq_len, tgt_seq_len]

    print(f"目标掩码形状: {tgt_mask.shape}")

    # 前向传播测试
    print("\n进行前向传播...")
    model.eval()  # 设置为评估模式
    with torch.no_grad():
        output = model(src, tgt, tgt_mask=tgt_mask)  # [batch_size, tgt_seq_len, tgt_vocab_size]

    print(f"输出形状: {output.shape}")
    print(f"输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")

    # 计算输出概率分布
    output_probs = F.softmax(output, dim=-1)  # [batch_size, tgt_seq_len, tgt_vocab_size]
    print(f"概率分布和 (应该接近1.0): {output_probs.sum(dim=-1)[0, 0].item():.6f}")

    # 测试训练模式
    print("\n测试训练模式...")
    model.train()  # 设置为训练模式

    # 创建目标标签（用于计算损失）
    tgt_labels = torch.randint(1, tgt_vocab_size, (batch_size, tgt_seq_len)).to(device)

    # 前向传播
    output = model(src, tgt, tgt_mask=tgt_mask)  # [batch_size, tgt_seq_len, tgt_vocab_size]

    # 计算交叉熵损失
    # 需要重塑张量以匹配损失函数要求
    output_flat = output.view(-1, tgt_vocab_size)  # [batch_size * tgt_seq_len, tgt_vocab_size]
    tgt_labels_flat = tgt_labels.view(-1)  # [batch_size * tgt_seq_len]

    loss = F.cross_entropy(output_flat, tgt_labels_flat)
    print(f"交叉熵损失: {loss.item():.4f}")

    # 测试反向传播
    print("测试反向传播...")
    loss.backward()

    # 检查梯度
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    print(f"模型参数有梯度: {has_grad}")

    # 内存使用情况（仅在CUDA可用时显示）
    if device.type == 'cuda':
        print(f"\nGPU内存使用:")
        print(f"  已分配: {torch.cuda.memory_allocated(device) / 1024 ** 2:.1f} MB")
        print(f"  已缓存: {torch.cuda.memory_reserved(device) / 1024 ** 2:.1f} MB")

    print("\n✅ Transformer模型测试完成！")


def test_individual_components():
    """测试各个组件的功能"""
    print("\n" + "=" * 50)
    print("测试各个组件...")
    print("=" * 50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 测试多头注意力
    print("\n1. 测试多头注意力机制...")
    d_model, n_heads = 512, 8
    seq_len, batch_size = 10, 4

    mha = MultiHeadAttention(d_model, n_heads).to(device)
    x = torch.randn(batch_size, seq_len, d_model).to(device)  # [batch_size, seq_len, d_model]

    # 自注意力测试
    attn_output = mha(x, x, x)  # [batch_size, seq_len, d_model]
    print(f"  输入形状: {x.shape}")
    print(f"  自注意力输出形状: {attn_output.shape}")
    print(f"  输出范围: [{attn_output.min().item():.4f}, {attn_output.max().item():.4f}]")

    # 测试带掩码的注意力
    mask = torch.ones(batch_size, seq_len, seq_len).to(device)  # [batch_size, seq_len, seq_len]
    mask[:, :, 5:] = 0  # 掩盖后半部分

    masked_output = mha(x, x, x, mask)  # [batch_size, seq_len, d_model]
    print(f"  带掩码的注意力输出形状: {masked_output.shape}")

    # 测试位置编码
    print("\n2. 测试位置编码...")
    pos_encoding = PositionalEncoding(d_model, max_len=100).to(device)
    x_with_pos = pos_encoding(x)  # [batch_size, seq_len, d_model]
    print(f"  位置编码后形状: {x_with_pos.shape}")
    print(f"  位置编码影响: {(x_with_pos - x).abs().mean().item():.6f}")

    # 测试前馈网络
    print("\n3. 测试位置前馈网络...")
    d_ff = 2048
    ffn = PositionwiseFeedForward(d_model, d_ff).to(device)
    ffn_output = ffn(x)  # [batch_size, seq_len, d_model]
    print(f"  前馈网络输出形状: {ffn_output.shape}")
    print(f"  输出范围: [{ffn_output.min().item():.4f}, {ffn_output.max().item():.4f}]")

    # 测试层归一化
    print("\n4. 测试层归一化...")
    layer_norm = LayerNorm(d_model).to(device)
    norm_output = layer_norm(x)  # [batch_size, seq_len, d_model]
    print(f"  层归一化输出形状: {norm_output.shape}")
    print(f"  归一化后均值: {norm_output.mean(dim=-1)[0, 0].item():.6f}")
    print(f"  归一化后标准差: {norm_output.std(dim=-1)[0, 0].item():.6f}")

    # 测试编码器层
    print("\n5. 测试编码器层...")
    encoder_layer = EncoderLayer(d_model, n_heads, d_ff).to(device)
    enc_output = encoder_layer(x)  # [batch_size, seq_len, d_model]
    print(f"  编码器层输出形状: {enc_output.shape}")

    # 测试解码器层
    print("\n6. 测试解码器层...")
    decoder_layer = DecoderLayer(d_model, n_heads, d_ff).to(device)
    # 创建编码器输出作为交叉注意力的输入
    encoder_out = torch.randn(batch_size, seq_len, d_model).to(device)  # [batch_size, seq_len, d_model]
    dec_output = decoder_layer(x, encoder_out)  # [batch_size, seq_len, d_model]
    print(f"  解码器层输出形状: {dec_output.shape}")

    print("\n✅ 所有组件测试完成！")


def test_demonstrate_attention_patterns():
    """展示注意力模式"""
    print("\n" + "=" * 50)
    print("展示注意力模式...")
    print("=" * 50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建简单的注意力层
    d_model, n_heads = 64, 4
    seq_len, batch_size = 8, 1

    mha = MultiHeadAttention(d_model, n_heads, dropout=0.0).to(device)
    mha.eval()  # 关闭dropout以获得确定性结果

    # 创建有模式的输入序列
    x = torch.randn(batch_size, seq_len, d_model).to(device)

    # 修改MultiHeadAttention类以返回注意力权重
    class AttentionVisualization(MultiHeadAttention):
        def forward(self, query, key, value, mask=None):
            batch_size, seq_len, d_model = query.shape

            Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
            K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
            V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

            # 计算注意力分数
            scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
            if mask is not None:
                mask = mask.unsqueeze(1)
                scores.masked_fill_(mask == 0, -1e9)

            # 获取注意力权重
            attention_weights = F.softmax(scores, dim=-1)

            # 计算输出
            attention_output = torch.matmul(attention_weights, V)
            attention_output = attention_output.transpose(1, 2).contiguous().view(
                batch_size, seq_len, self.d_model
            )
            output = self.w_o(attention_output)

            return output, attention_weights  # 返回输出和注意力权重

    # 创建可视化版本的注意力层
    vis_mha = AttentionVisualization(d_model, n_heads, dropout=0.0).to(device)
    vis_mha.load_state_dict(mha.state_dict())  # 复制权重
    vis_mha.eval()

    with torch.no_grad():
        output, attention_weights = vis_mha(x, x, x)

    print(f"注意力权重形状: {attention_weights.shape}")  # [batch_size, n_heads, seq_len, seq_len]

    # 显示第一个头的注意力模式
    first_head_attention = attention_weights[0, 0].cpu().numpy()  # [seq_len, seq_len]
    print("\n第一个注意力头的注意力矩阵:")
    print("(行=查询位置, 列=键位置)")
    for i in range(seq_len):
        row_str = " ".join([f"{first_head_attention[i, j]:.3f}" for j in range(seq_len)])
        print(f"位置{i}: {row_str}")

    # 验证注意力权重归一化
    attention_sums = attention_weights.sum(dim=-1)  # 应该全为1
    print(f"\n注意力权重和 (应该全为1): {attention_sums.mean().item():.6f} ± {attention_sums.std().item():.6f}")


def test_performance_benchmark():
    """性能基准测试"""
    print("\n" + "=" * 50)
    print("性能基准测试...")
    print("=" * 50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 不同规模的模型配置
    configs = [
        {"name": "Small", "d_model": 256, "n_heads": 4, "n_layers": 3, "d_ff": 1024},
        {"name": "Base", "d_model": 512, "n_heads": 8, "n_layers": 6, "d_ff": 2048},
        {"name": "Large", "d_model": 768, "n_heads": 12, "n_layers": 12, "d_ff": 3072},
    ]

    batch_size = 16
    seq_len = 128
    vocab_size = 10000

    for config in configs:
        print(f"\n测试 {config['name']} 模型:")
        print(f"  参数: d_model={config['d_model']}, n_heads={config['n_heads']}, "
              f"n_layers={config['n_layers']}, d_ff={config['d_ff']}")

        # 创建模型
        model = Transformer(
            src_vocab_size=vocab_size,
            tgt_vocab_size=vocab_size,
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            n_layers=config['n_layers'],
            d_ff=config['d_ff'],
            dropout=0.1
        ).to(device)

        # 计算参数数量
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  参数数量: {param_count:,}")

        # 创建测试数据
        src = torch.randint(1, vocab_size, (batch_size, seq_len)).to(device)
        tgt = torch.randint(1, vocab_size, (batch_size, seq_len)).to(device)
        tgt_mask = model.generate_square_subsequent_mask(seq_len, device)
        tgt_mask = tgt_mask.unsqueeze(0).expand(batch_size, -1, -1)

        # 预热
        model.train()
        for _ in range(3):
            output = model(src, tgt, tgt_mask=tgt_mask)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        # 测量前向传播时间
        import time
        start_time = time.time()
        num_iterations = 10

        for _ in range(num_iterations):
            output = model(src, tgt, tgt_mask=tgt_mask)
            if device.type == 'cuda':
                torch.cuda.synchronize()

        forward_time = (time.time() - start_time) / num_iterations
        print(f"  平均前向传播时间: {forward_time * 1000:.2f} ms")

        # 测量内存使用
        if device.type == 'cuda':
            memory_mb = torch.cuda.max_memory_allocated(device) / 1024 ** 2
            print(f"  峰值GPU内存使用: {memory_mb:.1f} MB")
            torch.cuda.reset_peak_memory_stats()

        # 清理
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()


def test_training_example():
    """简单的训练示例"""
    print("\n" + "=" * 50)
    print("简单训练示例...")
    print("=" * 50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建小型模型用于演示
    model = Transformer(
        src_vocab_size=1000,
        tgt_vocab_size=1000,
        d_model=256,
        n_heads=4,
        n_layers=2,
        d_ff=1024,
        dropout=0.1
    ).to(device)

    # 优化器 - 使用Adam优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)

    # 学习率调度器 - 论文中的学习率调度策略
    class TransformerLRScheduler:
        def __init__(self, optimizer, d_model, warmup_steps=4000):
            self.optimizer = optimizer
            self.d_model = d_model
            self.warmup_steps = warmup_steps
            self.step_num = 0

        def step(self):
            self.step_num += 1
            lr = self.d_model ** (-0.5) * min(self.step_num ** (-0.5),
                                              self.step_num * self.warmup_steps ** (-1.5))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr

    scheduler = TransformerLRScheduler(optimizer, model.encoder.d_model)

    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

    # 模拟训练数据
    batch_size = 16
    src_seq_len = 32
    tgt_seq_len = 32
    num_batches = 5

    model.train()

    for batch_idx in range(num_batches):
        # 生成随机数据（实际应用中这里是真实的训练数据）
        src = torch.randint(1, 1000, (batch_size, src_seq_len)).to(device)
        tgt_input = torch.randint(1, 1000, (batch_size, tgt_seq_len)).to(device)
        tgt_output = torch.randint(1, 1000, (batch_size, tgt_seq_len)).to(device)

        # 创建目标掩码
        tgt_mask = model.generate_square_subsequent_mask(tgt_seq_len, device)
        tgt_mask = tgt_mask.unsqueeze(0).expand(batch_size, -1, -1)

        # 前向传播
        output = model(src, tgt_input, tgt_mask=tgt_mask)  # [batch_size, tgt_seq_len, vocab_size]

        # 计算损失
        loss = F.cross_entropy(
            output.reshape(-1, output.size(-1)),  # [batch_size * tgt_seq_len, vocab_size]
            tgt_output.reshape(-1),  # [batch_size * tgt_seq_len]
            ignore_index=0  # 忽略填充token
        )

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 更新参数
        optimizer.step()
        lr = scheduler.step()

        print(f"批次 {batch_idx + 1}/{num_batches}: 损失={loss.item():.4f}, 学习率={lr:.2e}")

        if device.type == 'cuda':
            memory_mb = torch.cuda.memory_allocated(device) / 1024 ** 2
            print(f"  GPU内存使用: {memory_mb:.1f} MB")

    print("\n✅ 训练示例完成！")


if __name__ == "__main__":
    # 运行所有测试
    test_transformer()
    test_individual_components()
    test_demonstrate_attention_patterns()
    test_performance_benchmark()
    test_training_example()

    print("\n" + "=" * 50)
    print("🎉 所有测试完成！Transformer模型实现验证成功！")
    print("=" * 50)