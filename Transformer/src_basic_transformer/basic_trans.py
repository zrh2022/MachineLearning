# transformer_from_scratch_gpu.py  # 文件名说明，用于保存和运行的参考说明注释

import math  # 导入数学库，用于 sqrt、sin/cos 等运算
import time  # 导入时间库，用于计时训练耗时
import torch  # 导入 PyTorch 主库，用于张量和计算
import torch.nn as nn  # 从 PyTorch 导入神经网络模块别名 nn，用于定义模型层
import torch.optim as optim  # 导入优化器模块，用于训练优化器
from torch.utils.data import DataLoader, TensorDataset  # 导入数据加载工具，用于批量数据封装

# 设备选择：优先使用 GPU（cuda），否则回退到 CPU（cpu）  # 说明硬件选择策略
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 将设备对象赋值给 device 变量
print("使用设备：", device)  # 打印当前使用的设备，便于调试与确认
if device.type == "cuda":  # 如果当前设备为 CUDA
    torch.backends.cudnn.benchmark = True  # 启用 cudnn.benchmark 提升卷积等运算的性能（可选）

# -------------------- 超参数（可按需修改） --------------------  # 该行是分割说明，便于阅读
vocab_size = 10000  # 词表大小（示例用），实际任务请替换为 tokenizer 的 vocab_size
d_model = 512  # 模型隐藏维度，Embedding 与 Transformer 中的维度
num_heads = 8  # 多头注意力的头数
d_ff = 2048  # 前馈网络中间层维度（feed-forward hidden size）
num_encoder_layers = 3  # 编码器层数
num_decoder_layers = 3  # 解码器层数
dropout_rate = 0.1  # dropout 比率
max_seq_len = 64  # 最大序列长度（位置编码使用），训练时序列长度需 <= 该值
batch_size = 32  # 训练批次大小
lr = 1e-4  # 学习率
epochs = 3  # 训练轮数（示例用，真实训练请增大）

# -------------------- 位置编码（sinusoidal） --------------------  # 说明位置编码类型
class PositionalEncoding(nn.Module):  # 定义位置编码类，继承自 nn.Module
    def __init__(self, d_model, max_len=5000):  # 构造函数，接收模型维度和最大长度
        super().__init__()  # 调用父类构造函数初始化模块基类
        pe = torch.zeros(max_len, d_model)  # 创建位置编码矩阵，形状 (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # 位置索引列向量 (max_len,1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # 计算分母项用于 sin/cos 频率
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度使用 sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度使用 cos
        pe = pe.unsqueeze(0)  # 在最前面加 batch 维度，变为 (1, max_len, d_model)
        self.register_buffer('pe', pe)  # 注册位置编码为 buffer，随模型保存但不参与梯度更新

    def forward(self, x):  # 前向函数，接收输入 x，形状 (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :].to(x.device)  # 将位置编码切片并加到输入嵌入上，保持设备一致
        return x  # 返回带位置编码的嵌入

# -------------------- 缩放点积注意力函数 --------------------  # 说明内部具体实现函数
def scaled_dot_product_attention(q, k, v, attn_mask=None, dropout=None):  # 定义缩放点积注意力函数
    dk = q.size(-1)  # 获取 q 的最后一维大小，即每个 head 的维度（d_k）
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dk)  # 计算 QK^T / sqrt(dk) 得到注意力得分，形状 (B, h, T_q, T_k)
    if attn_mask is not None:  # 如果传入了注意力掩码
        # attn_mask 这里假定为布尔矩阵或可广播的张量，True 表示被 mask 的位置  # 说明掩码语义
        scores = scores.masked_fill(attn_mask.unsqueeze(1).unsqueeze(2), float("-inf"))  # 扩展掩码到 (B,1,1,T_k) 并填充 -inf
    attn = torch.softmax(scores, dim=-1)  # 对最后一个维度做 softmax 得到注意力权重
    if dropout is not None:  # 如果提供了 dropout 层
        attn = dropout(attn)  # 对注意力权重做 dropout
    output = torch.matmul(attn, v)  # 用注意力权重乘以 V 得到加权输出，形状 (B, h, T_q, d_k)
    return output, attn  # 返回输出与注意力权重

# -------------------- 多头注意力（手写实现） --------------------  # 说明接下来实现的类
class MultiHeadAttention(nn.Module):  # 定义多头注意力类，继承 nn.Module
    def __init__(self, d_model, num_heads, dropout=0.0):  # 构造函数，接收模型维度、头数和 dropout
        super().__init__()  # 调用父类构造函数
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"  # 确保 d_model 可整除 num_heads
        self.d_model = d_model  # 保存模型维度到实例变量
        self.num_heads = num_heads  # 保存头数到实例变量
        self.d_head = d_model // num_heads  # 每个 head 的维度（d_k）
        self.w_q = nn.Linear(d_model, d_model)  # 线性层用于生成 Q 的投影，输入 d_model，输出 d_model
        self.w_k = nn.Linear(d_model, d_model)  # 线性层用于生成 K 的投影，输入 d_model，输出 d_model
        self.w_v = nn.Linear(d_model, d_model)  # 线性层用于生成 V 的投影，输入 d_model，输出 d_model
        self.w_o = nn.Linear(d_model, d_model)  # 最后的线性投影层，用于把多头拼接后投回 d_model 维度
        self.dropout = nn.Dropout(dropout)  # dropout 层用于注意力权重上的随机失活

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):  # 前向函数，接收 Q/K/V 及掩码
        B = query.size(0)  # 批量大小 B
        # 线性投影并分头（reshape + transpose）: 先投影到 (B, T, d_model)
        q = self.w_q(query)  # Q 投影到 d_model 维度
        k = self.w_k(key)  # K 投影到 d_model 维度
        v = self.w_v(value)  # V 投影到 d_model 维度
        # 分割 heads：reshape 为 (B, T, num_heads, d_head) 然后转置为 (B, num_heads, T, d_head)
        q = q.view(B, -1, self.num_heads, self.d_head).transpose(1, 2)  # Q 分头形状变换
        k = k.view(B, -1, self.num_heads, self.d_head).transpose(1, 2)  # K 分头形状变换
        v = v.view(B, -1, self.num_heads, self.d_head).transpose(1, 2)  # V 分头形状变换
        # 构造注意力掩码：attn_mask 可以是 (T_q, T_k) 或 (B, T_q, T_k) 的布尔矩阵
        if key_padding_mask is not None:  # 如果存在 key_padding_mask（形状通常是 (B, T_k) 布尔）
            # 转换 key_padding_mask 为 (B, 1, 1, T_k) 以便广播到 scores 形状 (B, h, T_q, T_k)
            key_pad = key_padding_mask.unsqueeze(1).unsqueeze(2)  # 扩展维度以便广播掩码
            if attn_mask is None:  # 如果没有 attn_mask，则直接用 key_pad 作为总掩码
                total_mask = key_pad  # 将 key_pad 视为最终的掩码
            else:  # 如果同时有 attn_mask 与 key_padding_mask，需要合并
                total_mask = attn_mask.unsqueeze(0) | key_pad  # 按位或合并两个布尔掩码
        else:  # 如果没有 key_padding_mask
            total_mask = attn_mask  # 只有 attn_mask 被作为最终掩码
        # 调用缩放点积注意力函数，并传入 dropout 层
        attn_output, attn_weights = scaled_dot_product_attention(q, k, v, attn_mask=total_mask, dropout=self.dropout)  # 计算多头注意力输出
        # 将多头结果拼回：先 transpose 回 (B, T_q, num_heads, d_head)
        attn_output = attn_output.transpose(1, 2).contiguous()  # 转置回原始顺序并确保内存连续
        attn_output = attn_output.view(B, -1, self.d_model)  # 拼接多头，恢复为 (B, T_q, d_model)
        output = self.w_o(attn_output)  # 最后通过线性层投影回 d_model 维度
        return output, attn_weights  # 返回输出与注意力权重供外部使用或可视化

# -------------------- 前馈网络（逐位置前馈） --------------------  # 说明该子模块职责
class PositionwiseFeedForward(nn.Module):  # 定义逐位置前馈网络类
    def __init__(self, d_model, d_ff, dropout=0.0):  # 构造函数，接收模型维度、中间层维度和 dropout
        super().__init__()  # 调用父类构造函数
        self.fc1 = nn.Linear(d_model, d_ff)  # 第一个线性层，把 d_model 映射到 d_ff
        self.fc2 = nn.Linear(d_ff, d_model)  # 第二个线性层，把 d_ff 映射回 d_model
        self.activation = nn.ReLU()  # 激活函数采用 ReLU
        self.dropout = nn.Dropout(dropout)  # dropout 用于正则化

    def forward(self, x):  # 前向函数，接收输入 x 形状 (B, T, d_model)
        x = self.fc1(x)  # 线性映射到 d_ff 维度
        x = self.activation(x)  # 经过激活函数
        x = self.dropout(x)  # 经过 dropout 随机失活
        x = self.fc2(x)  # 再次线性映射回 d_model 维度
        return x  # 返回前馈网络输出

# -------------------- 编码器层（自注意力 + 前馈） --------------------  # 说明编码器层结构
class EncoderLayer(nn.Module):  # 定义单个编码器层，继承 nn.Module
    def __init__(self, d_model, num_heads, d_ff, dropout=0.0):  # 构造函数
        super().__init__()  # 调用父类构造函数
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)  # 实例化多头自注意力模块
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout=dropout)  # 实例化前馈网络模块
        self.norm1 = nn.LayerNorm(d_model)  # 第一个 LayerNorm（在 self-attn 之后）
        self.norm2 = nn.LayerNorm(d_model)  # 第二个 LayerNorm（在 ffn 之后）
        self.dropout1 = nn.Dropout(dropout)  # dropout1 用于 self-attn 输出
        self.dropout2 = nn.Dropout(dropout)  # dropout2 用于 ffn 输出

    def forward(self, src, src_mask=None, src_key_padding_mask=None):  # 前向函数，接收 src 与相应掩码
        attn_out, _ = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)  # 自注意力 Q=K=V=src
        src = src + self.dropout1(attn_out)  # 残差连接并加上 dropout 后与原始 src 相加
        src = self.norm1(src)  # 第一个层归一化（LayerNorm）
        ffn_out = self.ffn(src)  # 前馈网络计算
        src = src + self.dropout2(ffn_out)  # 残差连接并加上 dropout 后与前一步结果相加
        src = self.norm2(src)  # 第二个层归一化并返回
        return src  # 返回编码器层的输出

# -------------------- 解码器层（masked self-attn + enc-dec attn + ffn） --------------------  # 说明解码器层结构
class DecoderLayer(nn.Module):  # 定义单个解码器层，继承 nn.Module
    def __init__(self, d_model, num_heads, d_ff, dropout=0.0):  # 构造函数
        super().__init__()  # 调用父类构造函数
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)  # 解码器的 masked 自注意力模块
        self.enc_dec_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)  # 解码器到编码器的交叉注意力模块
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout=dropout)  # 逐位置前馈网络
        self.norm1 = nn.LayerNorm(d_model)  # 第一层 LayerNorm，位于 self-attn 之后
        self.norm2 = nn.LayerNorm(d_model)  # 第二层 LayerNorm，位于 enc-dec-attn 之后
        self.norm3 = nn.LayerNorm(d_model)  # 第三层 LayerNorm，位于 ffn 之后
        self.dropout1 = nn.Dropout(dropout)  # dropout1 对应 self-attn
        self.dropout2 = nn.Dropout(dropout)  # dropout2 对应 enc-dec-attn
        self.dropout3 = nn.Dropout(dropout)  # dropout3 对应 ffn

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):  # 前向函数
        # masked 自注意力（防止看到未来 token），Q=K=V=tgt
        self_attn_out, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)  # 计算 masked 自注意力
        tgt = tgt + self.dropout1(self_attn_out)  # 残差连接并 dropout
        tgt = self.norm1(tgt)  # LayerNorm1
        # 编码器-解码器注意力：Q=tgt, K=V=memory（来自 encoder 的输出）
        enc_dec_attn_out, _ = self.enc_dec_attn(tgt, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)  # 计算交叉注意力
        tgt = tgt + self.dropout2(enc_dec_attn_out)  # 残差连接并 dropout
        tgt = self.norm2(tgt)  # LayerNorm2
        ffn_out = self.ffn(tgt)  # 前馈网络
        tgt = tgt + self.dropout3(ffn_out)  # 残差连接并 dropout
        tgt = self.norm3(tgt)  # LayerNorm3
        return tgt  # 返回解码器层的输出

# -------------------- 编码器（多层堆叠） --------------------  # 说明编码器整体组成
class Encoder(nn.Module):  # 定义 Encoder 类，继承 nn.Module
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout):  # 构造函数
        super().__init__()  # 调用父类构造函数
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])  # 将多个编码器层封装为 ModuleList
        self.num_layers = num_layers  # 记录层数

    def forward(self, src, src_mask=None, src_key_padding_mask=None):  # 前向函数
        output = src  # 初始化输出为输入 src
        for layer in self.layers:  # 依次通过每一层编码器
            output = layer(output, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)  # 传递掩码与 padding mask
        return output  # 返回最终编码器输出

# -------------------- 解码器（多层堆叠） --------------------  # 说明解码器整体组成
class Decoder(nn.Module):  # 定义 Decoder 类，继承 nn.Module
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout):  # 构造函数
        super().__init__()  # 调用父类构造函数
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])  # 多个解码器层封装
        self.num_layers = num_layers  # 记录层数

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):  # 前向函数
        output = tgt  # 初始化输出为输入 tgt
        for layer in self.layers:  # 逐层处理
            output = layer(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)  # 调用每层
        return output  # 返回最终解码器输出

# -------------------- 完整的 Seq2Seq Transformer（Encoder-Decoder） --------------------  # 说明模型主体
class TransformerFromScratch(nn.Module):  # 定义整体 Transformer 模型类
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_encoder_layers=3, num_decoder_layers=3, d_ff=2048, dropout=0.1, max_seq_len=512):  # 构造函数
        super().__init__()  # 调用父类构造函数
        self.d_model = d_model  # 保存模型维度
        self.src_tok_emb = nn.Embedding(vocab_size, d_model)  # 源语言 token embedding 层
        self.tgt_tok_emb = nn.Embedding(vocab_size, d_model)  # 目标语言 token embedding 层
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)  # 位置编码器（用于 encoder）
        self.pos_decoder = PositionalEncoding(d_model, max_len=max_seq_len)  # 位置编码器（用于 decoder）
        self.encoder = Encoder(num_encoder_layers, d_model, num_heads, d_ff, dropout)  # 编码器堆栈
        self.decoder = Decoder(num_decoder_layers, d_model, num_heads, d_ff, dropout)  # 解码器堆栈
        self.generator = nn.Linear(d_model, vocab_size)  # 输出层（把 decoder 输出映射到词表大小）
        self._init_parameters()  # 初始化参数

    def _init_parameters(self):  # 参数初始化函数
        for p in self.parameters():  # 遍历所有参数张量
            if p.dim() > 1:  # 如果参数维度大于 1（即权重矩阵）
                nn.init.xavier_uniform_(p)  # 使用 Xavier 均匀分布初始化

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):  # 前向函数
        src_emb = self.src_tok_emb(src) * math.sqrt(self.d_model)  # 将 src token 映射为嵌入并缩放
        tgt_emb = self.tgt_tok_emb(tgt) * math.sqrt(self.d_model)  # 将 tgt token 映射为嵌入并缩放
        src_emb = self.pos_encoder(src_emb)  # 为 src 嵌入添加位置编码
        tgt_emb = self.pos_decoder(tgt_emb)  # 为 tgt 嵌入添加位置编码
        memory = self.encoder(src_emb, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)  # 编码器处理得到 memory
        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask, memory_mask=None, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)  # 解码器处理
        logits = self.generator(output)  # 通过线性层得到词表 logits，形状 (B, T, V)
        return logits  # 返回 logits 供损失计算或解码使用

# -------------------- 掩码生成函数 --------------------  # 说明接下来是掩码辅助函数
def generate_square_subsequent_mask(sz):  # 定义生成下三角的 subsequent mask 函数
    # 返回形状 (sz, sz) 的布尔矩阵，True 表示被 mask（即禁止注意到未来）
    mask = torch.triu(torch.ones(sz, sz, dtype=torch.bool), diagonal=1)  # 生成上三角（不含对角线）作为 mask
    return mask  # 返回布尔类型的掩码

def create_padding_mask(seq, pad_idx=0):  # 创建 padding mask，seq 形状 (B, T)，pad_idx 表示 padding 的 id
    # 返回形状 (B, T) 的布尔张量，True 表示该位置为填充（应被 mask）
    return (seq == pad_idx)  # 等于 pad_idx 的位置返回 True

# -------------------- 训练示例（使用随机数据做演示） --------------------  # 说明示例目的
def example_train():  # 定义示例训练函数
    # 准备随机数据作为示例：真实任务请替换为真实 Dataset 与 tokenizer 结果
    num_samples = 200  # 样本数量较小，仅用于演示
    seq_len = max_seq_len  # 使用全局最大序列长度作为示例序列长度
    src_data = torch.randint(1, vocab_size, (num_samples, seq_len), dtype=torch.long)  # 随机生成 src token id，避免 0（pad）
    tgt_data = torch.randint(1, vocab_size, (num_samples, seq_len), dtype=torch.long)  # 随机生成 tgt token id，避免 0（pad）
    # 随机把一些位置设为 pad（0），模拟不同长度样本（这里概率为 10%）
    mask_positions = torch.rand(src_data.shape) < 0.1  # 生成布尔矩阵，True 表示该位置需要被设为 pad
    src_data[mask_positions] = 0  # 将 src 中的这些位置设为 pad id 0
    mask_positions = torch.rand(tgt_data.shape) < 0.1  # 为 tgt 也生成随机 pad
    tgt_data[mask_positions] = 0  # 将 tgt 中的这些位置设为 pad id 0
    dataset = TensorDataset(src_data, tgt_data)  # 用 TensorDataset 封装数据
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # 用 DataLoader 生成批次并打乱数据

    model = TransformerFromScratch(vocab_size=vocab_size, d_model=d_model, num_heads=num_heads, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, d_ff=d_ff, dropout=dropout_rate, max_seq_len=max_seq_len)  # 实例化模型
    model = model.to(device)  # 将模型移动到指定设备（GPU 或 CPU）
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 交叉熵损失，忽略 pad id（0）
    optimizer = optim.Adam(model.parameters(), lr=lr)  # 使用 Adam 优化器

    model.train()  # 将模型设为训练模式
    for epoch in range(1, epochs + 1):  # 训练若干轮次
        epoch_loss = 0.0  # 用于累积该轮的损失
        t0 = time.time()  # 记录该轮开始时间
        for batch_idx, (src_batch, tgt_batch) in enumerate(dataloader):  # 遍历每个批次
            src_batch = src_batch.to(device)  # 将 src 批次移动到设备
            tgt_batch = tgt_batch.to(device)  # 将 tgt 批次移动到设备
            tgt_input = tgt_batch[:, :-1]  # 解码器输入通常是目标序列右移一位（去掉最后一个 token）
            tgt_output = tgt_batch[:, 1:]  # 解码器预测目标是目标序列左移一位（去掉第一个 token）
            # 生成掩码：target 的 subsequent mask（下三角）用于防止看到未来
            tgt_mask = generate_square_subsequent_mask(tgt_input.size(1)).to(device)  # 生成下三角掩码并移动到设备
            # 生成 padding mask：True 表示该位置为 padding，应在 attention 中被忽略
            src_key_padding_mask = create_padding_mask(src_batch, pad_idx=0).to(device)  # src 的 padding mask
            tgt_key_padding_mask = create_padding_mask(tgt_input, pad_idx=0).to(device)  # tgt 的 padding mask
            memory_key_padding_mask = src_key_padding_mask  # memory padding mask 与 src padding mask 相同

            optimizer.zero_grad()  # 梯度清零
            logits = model(src_batch, tgt_input, src_mask=None, tgt_mask=tgt_mask, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)  # 前向计算得到 logits，形状 (B, T, V)
            logits_flat = logits.reshape(-1, logits.size(-1))  # 将 logits 展平成 (B*T, V) 以便计算交叉熵
            tgt_flat = tgt_output.reshape(-1)  # 将目标展平成 (B*T,)
            loss = criterion(logits_flat, tgt_flat)  # 计算损失，自动忽略 pad id
            loss.backward()  # 反向传播计算梯度
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪以避免梯度爆炸
            optimizer.step()  # 优化器更新参数

            epoch_loss += loss.item()  # 累加损失值用于统计
        t1 = time.time()  # 记录该轮结束时间
        print(f"Epoch {epoch} | loss={epoch_loss/len(dataloader):.6f} | time={(t1-t0):.2f}s")  # 打印该轮平均损失与耗时

    # 保存模型参数到文件，便于后续加载或推理使用
    torch.save(model.state_dict(), "transformer_from_scratch.pth")  # 将模型参数保存到本地文件
    print("训练完成，模型已保存： transformer_from_scratch.pth")  # 打印保存成功提示
    return model  # 返回训练好的模型实例

# -------------------- 贪心解码（单条样本） --------------------  # 说明解码策略
@torch.no_grad()  # 在推理时不需要计算梯度，加装饰器避免多余计算
def greedy_decode(model, src, max_len, start_symbol=1):  # 定义贪心解码函数，接收模型、src、最大长度与起始符 id
    model.eval()  # 将模型设置为评估模式（关闭 dropout 等）
    src = src.to(device)  # 将 src 移动到设备
    src_key_padding_mask = create_padding_mask(src, pad_idx=0).to(device)  # 生成 src 的 padding mask
    # encoder 端得到 memory
    src_emb = model.src_tok_emb(src) * math.sqrt(model.d_model)  # 将 src 编码为嵌入并缩放
    src_emb = model.pos_encoder(src_emb)  # 添加位置编码
    memory = model.encoder(src_emb, src_mask=None, src_key_padding_mask=src_key_padding_mask)  # 得到 memory 张量
    ys = torch.ones(1, 1, dtype=torch.long).to(device) * start_symbol  # 初始化解码序列，batch=1，起始符 start_symbol
    for i in range(max_len - 1):  # 循环生成直到达到最大长度
        tgt_mask = generate_square_subsequent_mask(ys.size(1)).to(device)  # 为当前已生成序列构造 subsequent mask
        tgt_key_padding_mask = create_padding_mask(ys, pad_idx=0).to(device)  # tgt 的 padding mask（若有 pad）
        logits = model(None, None)  # 占位调用避免 lint 报错（下面会重写实现），此行会被下一行实际替代逻辑覆盖
        # 实际上我们不能通过 model(src, ys) 因为前向函数要求传入 src 和 tgt 都非 None，所以下面单独复用 encoder 和 decoder 逻辑
        tgt_emb = model.tgt_tok_emb(ys) * math.sqrt(model.d_model)  # 将当前已生成的 ys 映射为嵌入并缩放
        tgt_emb = model.pos_decoder(tgt_emb)  # 为 tgt_emb 添加位置编码
        out = model.decoder(tgt_emb, memory, tgt_mask=tgt_mask, memory_mask=None, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=src_key_padding_mask)  # 调用 decoder 得到输出
        prob = model.generator(out[:, -1, :])  # 取最后一个时间步的 logits 作为下一个 token 的分布
        next_word = torch.argmax(prob, dim=-1).unsqueeze(1)  # 贪心选取概率最高的 token，扩展成 (B,1)
        ys = torch.cat([ys, next_word], dim=1)  # 将新生成的 token 拼接到 ys 后面
    return ys  # 返回生成的 token id 序列

# -------------------- 主入口（运行训练与演示解码） --------------------  # 说明脚本主流程
if __name__ == "__main__":  # 仅当脚本作为主程序运行时执行以下内容
    model = example_train()  # 调用示例训练函数并获取训练好的模型
    # 用训练好的模型做一次贪心解码演示（构造随机 src），真实使用时请用 tokenizer 得到的 token ids
    src_example = torch.randint(1, vocab_size, (1, max_seq_len), dtype=torch.long)  # 随机构造 1 条 src 示例
    # 随机把一些位置设为 pad id 0，模拟真实输入（可选）
    pad_mask = torch.rand(src_example.shape) < 0.1  # 随机生成 pad 掩码
    src_example[pad_mask] = 0  # 将这些位置设为 0（pad）
    decoded = greedy_decode(model, src_example, max_len=30, start_symbol=1)  # 使用贪心解码得到生成序列
    print("解码结果（token ids）:", decoded.tolist())  # 打印解码得到的 token id 序列供观察
