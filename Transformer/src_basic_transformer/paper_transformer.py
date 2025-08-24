# 下面的代码实现了论文 "Attention Is All You Need"（Transformer）的核心结构与训练细节，包含论文中常见的优化技巧：残差连接、LayerNorm、Dropout、多头自注意力、位置编码、前馈网络、Adam+Noam学习率调度、标签平滑、梯度裁剪、权重初始化等。每一行非空白代码行都写有注释（按用户要求）。
# 代码使用 PyTorch 实现，尽量使用基础层（Linear、LayerNorm、Embedding 等），没有使用 nn.Transformer 之类的高阶封装；支持 GPU（如果可用）。
# 最后包含一个简单的案例：序列拷贝任务（输入序列 -> 输出相同序列），用于演示模型训练与推理流程。
import math  # 数学函数和常量的支持
import copy  # 用于深拷贝模块参数
import time  # 记录训练时间
import random  # 用于可重复性设置的随机数
from typing import Optional  # 类型提示，可选参数
import torch  # PyTorch 主库
import torch.nn as nn  # PyTorch 神经网络模块
import torch.nn.functional as F  # PyTorch 常用函数
from torch.utils.data import Dataset, DataLoader  # 数据处理工具

# ----------------------------- 基础工具函数与超参 -----------------------------
# 固定随机种子以便复现实验结果
def set_seed(seed: int = 42):  # 设置随机种子的函数
    random.seed(seed)  # Python 随机
    torch.manual_seed(seed)  # CPU 随机
    if torch.cuda.is_available():  # 如果可用 GPU
        torch.cuda.manual_seed_all(seed)  # 设置 GPU 随机

set_seed(42)  # 设定全局随机种子，保证可复现

# 设备选择（GPU 优先）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 选择运行设备

# ----------------------------- 位置编码 -----------------------------
class PositionalEncoding(nn.Module):  # 位置编码模块，继承 nn.Module
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):  # 初始化
        super().__init__()  # 调用父类构造函数
        self.dropout = nn.Dropout(p=dropout)  # 使用 dropout 减少过拟合
        # 创建位置编码矩阵，形状为 (max_len, d_model)
        pe = torch.zeros(max_len, d_model)  # 初始化全零位置编码矩阵
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # 位置索引 (max_len,1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # 频率项
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维使用 sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维使用 cos
        pe = pe.unsqueeze(0).transpose(0, 1)  # 变为 (max_len, 1, d_model)
        self.register_buffer("pe", pe)  # 注册为 buffer，随模型保存但不作为参数

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 前向传播，x: (seq_len, batch, d_model)
        x = x + self.pe[: x.size(0), :]  # 加上位置编码
        return self.dropout(x)  # 返回加了位置编码并经过 dropout 的张量

# ----------------------------- 注意力与多头注意力 -----------------------------
def clones(module: nn.Module, N: int) -> nn.Module:  # 复制模块 N 次（用于堆叠层）
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])  # 返回 ModuleList

def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor], dropout: Optional[nn.Module]):
    # scaled dot-product attention 计算函数
    # query, key, value: (batch, head, seq_len, d_k) —— 注意这里我们在实现 MultiHead 时会调整维度
    d_k = query.size(-1)  # d_k 是每个头的维度
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # 计算 QK^T 并缩放
    if mask is not None:  # 如果提供了 mask（比如解码器的未来信息屏蔽）
        scores = scores.masked_fill(mask == 0, -1e9)  # 对被 mask 的位置赋很小的值
    p_attn = F.softmax(scores, dim=-1)  # softmax 获得注意力分布
    if dropout is not None:  # 如果有 dropout
        p_attn = dropout(p_attn)  # 对注意力分布进行 dropout
    return torch.matmul(p_attn, value), p_attn  # 返回注意力加权后的值与注意力权重

class MultiHeadedAttention(nn.Module):  # 多头注意力模块
    def __init__(self, h: int, d_model: int, dropout: float = 0.1):  # 初始化，h 为头数
        super().__init__()  # 调用父类构造函数
        assert d_model % h == 0, "d_model must be divisible by h"  # 确保可均分
        self.d_k = d_model // h  # 每个头的维度
        self.h = h  # 保存头数
        # 为 Q, K, V 各自定义一个线性映射（使用基础层 Linear）
        self.linears = clones(nn.Linear(d_model, d_model), 4)  # 4 个线性：Q、K、V、输出
        self.attn = None  # 存储注意力矩阵以便调试/可视化
        self.dropout = nn.Dropout(p=dropout)  # 在注意力权重上使用 dropout

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None):
        if mask is not None:  # mask 最初形状可能是 (batch, 1, seq_len) 或类似
            mask = mask.unsqueeze(1)  # 扩展到 (batch, 1, 1, seq_len) 以匹配 attention 记号
        nbatches = query.size(0)  # batch 大小

        # 1) 通过线性层并分割为多个头
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]  # 得到 (batch, head, seq_len, d_k)

        # 2) 应用 attention
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)  # 得到 (batch, head, seq_len, d_k)

        # 3) 组合 heads 并通过最终线性层
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)  # 合并为 (batch, seq_len, d_model)
        return self.linears[-1](x)  # 通过最后一个线性层映射回 d_model 维度

# ----------------------------- 前馈网络 -----------------------------
class PositionwiseFeedForward(nn.Module):  # 位置前馈网络
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):  # 初始化
        super().__init__()  # 父类构造
        self.w_1 = nn.Linear(d_model, d_ff)  # 第一层线性，扩展到 d_ff
        self.w_2 = nn.Linear(d_ff, d_model)  # 第二层线性，投回 d_model
        self.dropout = nn.Dropout(p=dropout)  # dropout 减少过拟合

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 前向传播
        return self.w_2(self.dropout(F.relu(self.w_1(x))))  # w1->ReLU->dropout->w2

# ----------------------------- 残差连接 + 层归一化 -----------------------------
class SublayerConnection(nn.Module):  # Sublayer + residual + LayerNorm
    def __init__(self, size: int, dropout: float):  # 初始化
        super().__init__()  # 父类构造
        self.norm = nn.LayerNorm(size)  # 使用 LayerNorm 而非 BatchNorm（论文推荐）
        self.dropout = nn.Dropout(p=dropout)  # dropout

    def forward(self, x: torch.Tensor, sublayer):  # sublayer 是一个函数/模块
        return x + self.dropout(sublayer(self.norm(x)))  # 残差连接：x + Dropout(sublayer(LN(x)))

# ----------------------------- Encoder 与 Decoder 层 -----------------------------
class EncoderLayer(nn.Module):  # Encoder 层：包含多头注意力和前馈网络（各自带残差与LayerNorm）
    def __init__(self, size: int, self_attn: MultiHeadedAttention, feed_forward: PositionwiseFeedForward, dropout: float):
        super().__init__()  # 父类构造
        self.self_attn = self_attn  # 自注意力模块
        self.feed_forward = feed_forward  # 前馈模块
        self.sublayer = clones(SublayerConnection(size, dropout), 2)  # 两个 sublayer：attention 和 feed-forward
        self.size = size  # d_model

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:  # 前向
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))  # 自注意力子层
        return self.sublayer[1](x, self.feed_forward)  # 前馈子层

class DecoderLayer(nn.Module):  # Decoder 层：包含自注意力、Encoder-Decoder注意力、前馈网络
    def __init__(self, size: int, self_attn: MultiHeadedAttention, src_attn: MultiHeadedAttention, feed_forward: PositionwiseFeedForward, dropout: float):
        super().__init__()  # 父类构造
        self.size = size  # d_model
        self.self_attn = self_attn  # decoder 自注意力
        self.src_attn = src_attn  # encoder-decoder 注意力
        self.feed_forward = feed_forward  # 前馈网络
        self.sublayer = clones(SublayerConnection(size, dropout), 3)  # 三个子层：self-attn, src-attn, ff

    def forward(self, x: torch.Tensor, memory: torch.Tensor, src_mask: Optional[torch.Tensor], tgt_mask: Optional[torch.Tensor]) -> torch.Tensor:
        m = memory  # encoder 输出记为 m
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))  # decoder 自注意力（masked）
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))  # encoder-decoder 注意力
        return self.sublayer[2](x, self.feed_forward)  # 前馈子层

# ----------------------------- Encoder 与 Decoder 堆叠 -----------------------------
class Encoder(nn.Module):  # 完整的 Encoder，由 N 层 EncoderLayer 堆叠而成
    def __init__(self, layer: EncoderLayer, N: int):
        super().__init__()  # 父类构造
        self.layers = clones(layer, N)  # N 层复制
        self.norm = nn.LayerNorm(layer.size)  # 最后再加一个 LayerNorm（论文实现细节）

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:  # 前向
        for layer in self.layers:  # 逐层
            x = layer(x, mask)  # 通过每一层
        return self.norm(x)  # 最后归一化后返回

class Decoder(nn.Module):  # 完整的 Decoder，由 N 层 DecoderLayer 堆叠而成
    def __init__(self, layer: DecoderLayer, N: int):
        super().__init__()  # 父类构造
        self.layers = clones(layer, N)  # N 层复制
        self.norm = nn.LayerNorm(layer.size)  # 最后 LayerNorm

    def forward(self, x: torch.Tensor, memory: torch.Tensor, src_mask: Optional[torch.Tensor], tgt_mask: Optional[torch.Tensor]) -> torch.Tensor:
        for layer in self.layers:  # 逐层
            x = layer(x, memory, src_mask, tgt_mask)  # 通过每一层
        return self.norm(x)  # 返回归一化输出

# ----------------------------- 完整 Transformer 模型构建器 -----------------------------
class Transformer(nn.Module):  # 最终的 Encoder-Decoder Transformer
    def __init__(self, src_vocab: int, tgt_vocab: int, N: int = 2, d_model: int = 128, d_ff: int = 512, h: int = 8, dropout: float = 0.1):
        super().__init__()  # 父类构造
        # 词嵌入层（源与目标）并可选共享（论文中源/目标和输出 embedding 可能共享）
        self.src_embed = nn.Sequential(nn.Embedding(src_vocab, d_model), PositionalEncoding(d_model, dropout=dropout))  # 源嵌入 + 位置编码
        self.tgt_embed = nn.Sequential(nn.Embedding(tgt_vocab, d_model), PositionalEncoding(d_model, dropout=dropout))  # 目标嵌入 + 位置编码
        # 构建基本模块：多头注意力、前馈网络、EncoderLayer、DecoderLayer
        attn = MultiHeadedAttention(h, d_model, dropout)  # 多头注意力实例
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)  # 前馈网络实例
        # Encoder 与 Decoder 的层
        encoder_layer = EncoderLayer(d_model, copy.deepcopy(attn), copy.deepcopy(ff), dropout)  # Encoder 层
        decoder_layer = DecoderLayer(d_model, copy.deepcopy(attn), copy.deepcopy(attn), copy.deepcopy(ff), dropout)  # Decoder 层
        # Encoder 与 Decoder 堆叠
        self.encoder = Encoder(encoder_layer, N)  # N 层 Encoder
        self.decoder = Decoder(decoder_layer, N)  # N 层 Decoder
        # 最后的输出线性层，将 d_model 映射到目标词表大小
        self.out = nn.Linear(d_model, tgt_vocab)  # 输出映射层

        # 权重初始化（论文中建议合适的初始化），这里使用 xavier_uniform 初始化线性层权重
        for p in self.parameters():  # 遍历所有参数
            if p.dim() > 1:  # 仅对权重矩阵使用 xavier 初始化
                nn.init.xavier_uniform_(p)  # Xavier 均匀初始化

    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor]) -> torch.Tensor:  # 编码过程
        return self.encoder(self.src_embed(src), src_mask)  # 嵌入 + 编码器堆栈

    def decode(self, memory: torch.Tensor, src_mask: Optional[torch.Tensor], tgt: torch.Tensor, tgt_mask: Optional[torch.Tensor]) -> torch.Tensor:
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)  # 嵌入 + 解码器堆栈

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: Optional[torch.Tensor], tgt_mask: Optional[torch.Tensor]) -> torch.Tensor:
        memory = self.encode(src, src_mask)  # 得到 encoder 输出 memory
        dec = self.decode(memory, src_mask, tgt, tgt_mask)  # 得到 decoder 输出
        return self.out(dec)  # 通过输出映射得到 logits

# ----------------------------- Mask 创建器 -----------------------------
def make_std_mask(tgt: torch.Tensor, pad: int):  # 创建目标 mask（屏蔽 pad 与未来信息）
    tgt_mask = (tgt != pad).unsqueeze(-2)  # 屏蔽 pad 的位置，形状扩展为 (batch,1,seq_len)
    seq_len = tgt.size(1)  # 目标序列长度
    subsequent_mask = torch.triu(torch.ones((1, seq_len, seq_len), device=tgt.device), diagonal=1).type(torch.uint8)  # 上三角掩码（未来屏蔽）
    tgt_mask = tgt_mask & (subsequent_mask == 0)  # 同时屏蔽 pad 和未来位置
    return tgt_mask  # 返回 bool 类型的 mask

def make_src_mask(src: torch.Tensor, pad: int):  # 创建源 mask（屏蔽 pad）
    return (src != pad).unsqueeze(-2)  # 返回 (batch,1,seq_len) 的 mask

# ----------------------------- 标签平滑损失 -----------------------------
class LabelSmoothingLoss(nn.Module):  # 标签平滑实现
    def __init__(self, size: int, padding_idx: int, smoothing: float = 0.1):  # 初始化
        super().__init__()  # 父类构造
        self.criterion = nn.KLDivLoss(reduction='sum')  # 使用 KLDivLoss 计算平滑后的分布差异
        self.padding_idx = padding_idx  # pad 索引（不计算损失）
        self.confidence = 1.0 - smoothing  # 真实标签信心
        self.smoothing = smoothing  # 平滑系数
        self.size = size  # 词表大小

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:  # x: (batch*seq_len, vocab), target: (batch*seq_len)
        assert x.size(1) == self.size  # 维度检查
        true_dist = x.data.clone()  # 克隆 x 的数据形状，准备生成目标分布
        true_dist.fill_(self.smoothing / (self.size - 2))  # 除去正确标签与 pad 所分配的概率
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)  # 在正确标签上放置信心
        true_dist[:, self.padding_idx] = 0  # pad 的位置概率设为 0
        mask = torch.nonzero(target.data == self.padding_idx)  # 找出 pad 的位置索引
        if mask.dim() > 0:  # 如果有 pad
            true_dist.index_fill_(0, mask.squeeze(), 0.0)  # 将 pad 行置零
        return self.criterion(F.log_softmax(x, dim=1), true_dist)  # 返回 KL 散度损失总和

# ----------------------------- 学习率调度器（Noam） -----------------------------
class NoamOpt:  # Noam 学习率调度器（论文中使用）
    def __init__(self, model_size: int, factor: float, warmup: int, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer  # 外部传入的优化器
        self._step = 0  # 记录步数
        self.warmup = warmup  # warmup 步数
        self.factor = factor  # 因子
        self.model_size = model_size  # 模型尺寸 d_model
        self._rate = 0  # 当前学习率

    def step(self):  # 在每次参数更新后调用
        self._step += 1  # 计数步数
        rate = self.rate()  # 计算当前 lr
        for p in self.optimizer.param_groups:  # 更新所有参数组的 lr
            p['lr'] = rate  # 设定 lr
        self._rate = rate  # 存储当前 lr
        self.optimizer.step()  # 调用 optimizer.step()

    def rate(self, step: Optional[int] = None) -> float:  # 计算学习率函数
        if step is None:  # 默认使用当前步数
            step = self._step  # 使用内部记录
        return self.factor * (self.model_size ** (-0.5) *
                              min(step ** (-0.5), step * self.warmup ** (-1.5)))  # Noam 公式

# ----------------------------- 简单数据集（序列拷贝任务） -----------------------------
class CopyDataset(Dataset):  # 简单的拷贝任务数据集，用于 demo
    def __init__(self, vocab_size: int, seq_len: int, dataset_size: int, pad_idx: int):
        super().__init__()  # 父类构造
        self.vocab_size = vocab_size  # 词表大小
        self.seq_len = seq_len  # 序列长度
        self.dataset_size = dataset_size  # 数据集大小
        self.pad_idx = pad_idx  # pad 索引
        self.data = []  # 存放数据

        for _ in range(dataset_size):  # 生成 dataset_size 个样本
            length = random.randint(1, seq_len)  # 随机真实长度
            seq = [random.randint(2, vocab_size - 1) for _ in range(length)]  # 随机 token，避开 0/1 保留给特殊符号
            src = seq + [pad_idx] * (seq_len - len(seq))  # 源端 pad 填充到固定长度
            tgt = seq + [pad_idx] * (seq_len - len(seq))  # 目标端相同（拷贝任务）
            self.data.append((torch.LongTensor(src), torch.LongTensor(tgt)))  # 存入数据

    def __len__(self):  # 返回数据集大小
        return self.dataset_size  # 长度

    def __getitem__(self, idx: int):  # 返回指定样本
        return self.data[idx]  # 返回 (src, tgt)

# ----------------------------- 训练与评估函数 -----------------------------
def run_epoch(data_loader: DataLoader, model: Transformer, criterion: nn.Module, optimizer: NoamOpt, pad_idx: int, epoch: int, clip: float = 1.0):
    model.train()  # 训练模式
    total_loss = 0.0  # 累计损失
    start = time.time()  # 记录起始时间
    for i, (src, tgt) in enumerate(data_loader):  # 遍历数据
        src = src.to(device)  # 将源数据搬到设备
        tgt = tgt.to(device)  # 将目标数据搬到设备
        # 构建输入输出，target_in 为解码器的输入（左移一位，并以 BOS=1 填充）
        tgt_input = torch.cat([torch.ones(src.size(0), 1, dtype=torch.long, device=device), tgt[:, :-1]], dim=1)  # 在最前面加 BOS
        src_mask = make_src_mask(src, pad_idx)  # 源 mask
        tgt_mask = make_std_mask(tgt_input, pad_idx)  # 目标 mask（包含未来遮蔽）
        logits = model(src, tgt_input, src_mask, tgt_mask)  # 前向得到 logits，形状 (batch, seq_len, vocab)
        # 计算损失：平滑后的标签
        loss = criterion(logits.view(-1, logits.size(-1)), tgt.view(-1))  # 计算标签平滑损失（KLDivLoss 返回总和）
        optimizer.optimizer.zero_grad()  # 清零原始 optimizer 的梯度
        loss.backward()  # 反向传播计算梯度
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  # 梯度裁剪，防止梯度爆炸
        optimizer.step()  # 更新参数并按 Noam 调度学习率
        total_loss += loss.item()  # 累计损失数值
        if (i + 1) % 10 == 0:  # 每 10 个 batch 打印一次信息
            elapsed = time.time() - start  # 计算耗时
            print(f"Epoch {epoch} | Batch {i+1}/{len(data_loader)} | Loss {loss.item():.4f} | Time {elapsed:.2f}s")  # 输出训练信息
            start = time.time()  # 重置计时器
    return total_loss / len(data_loader.dataset)  # 返回每个样本平均损失（总和除以样本数）

def evaluate(model: Transformer, data_loader: DataLoader, pad_idx: int):
    model.eval()  # 评估模式
    total_correct = 0  # 正确计数
    total_tokens = 0  # 总 token 数
    with torch.no_grad():  # 评估时不计算梯度
        for src, tgt in data_loader:  # 遍历数据
            src = src.to(device)  # 转移到设备
            tgt = tgt.to(device)  # 转移到设备
            # 逐步贪婪解码（演示用，非 beam search）
            batch_size, seq_len = src.size(0), tgt.size(1)  # 获取 batch 和序列长度
            memory = model.encode(src, make_src_mask(src, pad_idx))  # 编码器输出
            ys = torch.ones(batch_size, 1, dtype=torch.long, device=device)  # 初始解码输入：BOS
            for i in range(seq_len):  # 逐步生成
                out = model.decode(memory, make_src_mask(src, pad_idx), ys, make_std_mask(ys, pad_idx))  # 解码器输出
                prob = F.softmax(model.out(out[:, -1]), dim=-1)  # 对最后一步做 softmax 获得概率
                _, next_word = torch.max(prob, dim=1)  # 取概率最大词
                ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)  # 将生成词拼接到 ys
            pred = ys[:, 1:seq_len+1]  # 丢弃 BOS，取 seq_len 个 token 作为预测
            mask = (tgt != pad_idx)  # 忽略 pad 的位置
            total_correct += torch.sum((pred == tgt) & mask).item()  # 累计正确 token 数
            total_tokens += torch.sum(mask).item()  # 累计有效 token 数
    return total_correct / total_tokens if total_tokens > 0 else 0.0  # 返回 token-level 精度

# ----------------------------- 小规模示例训练脚本 -----------------------------
def demo_train():
    # 超参数设置（为 demo 缩小模型规模以便快速运行）
    SRC_VOCAB = 50  # 源词表大小（含特殊符号）
    TGT_VOCAB = 50  # 目标词表大小
    PAD_IDX = 0  # pad 索引
    BOS_IDX = 1  # bos 索引
    SEQ_LEN = 10  # 序列固定长度
    DATASET_SIZE = 200  # 数据集样本数
    BATCH_SIZE = 16  # batch 大小
    N_EPOCHS = 3  # 训练轮数（示例中少量迭代）
    D_MODEL = 128  # 模型隐层维度
    D_FF = 512  # 前馈网络维度
    H = 8  # 注意力头数
    N_LAYERS = 2  # encoder/decoder 层数
    DROPOUT = 0.1  # dropout 比例

    dataset = CopyDataset(SRC_VOCAB, SEQ_LEN, DATASET_SIZE, PAD_IDX)  # 创建拷贝任务数据集
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)  # DataLoader

    model = Transformer(SRC_VOCAB, TGT_VOCAB, N=N_LAYERS, d_model=D_MODEL, d_ff=D_FF, h=H, dropout=DROPOUT).to(device)  # 初始化模型并转到设备
    criterion = LabelSmoothingLoss(size=TGT_VOCAB, padding_idx=PAD_IDX, smoothing=0.1)  # 标签平滑损失
    # 使用 Adam 优化器并传入 Noam 学习率调度器（论文的设置）
    adam = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)  # 初始化 Adam，lr 由 Noam 控制
    optimizer = NoamOpt(D_MODEL, factor=1, warmup=200, optimizer=adam)  # Noam 调度

    # 训练循环
    for epoch in range(1, N_EPOCHS + 1):  # 多轮训练
        avg_loss = run_epoch(data_loader, model, criterion, optimizer, PAD_IDX, epoch, clip=1.0)  # 运行一轮训练
        acc = evaluate(model, data_loader, PAD_IDX)  # 在训练集上评估准确率（演示用）
        print(f"Epoch {epoch} finished. Avg Loss per sample {avg_loss:.6f}. Token Accuracy {acc*100:.2f}%")  # 打印结果

    # 演示单条推理（显示一个样本的输入与输出）
    model.eval()  # 切换到评估模式
    sample_src, sample_tgt = dataset[0]  # 取第一个样本
    sample_src = sample_src.unsqueeze(0).to(device)  # 增加 batch 维度并转到设备
    memory = model.encode(sample_src, make_src_mask(sample_src, PAD_IDX))  # 编码
    ys = torch.ones(1, 1, dtype=torch.long, device=device)  # 初始解码输入 BOS
    for i in range(SEQ_LEN):  # 逐步生成
        out = model.decode(memory, make_src_mask(sample_src, PAD_IDX), ys, make_std_mask(ys, PAD_IDX))  # 解码器输出
        prob = F.softmax(model.out(out[:, -1]), dim=-1)  # 最后一时刻概率
        _, next_word = torch.max(prob, dim=1)  # 选取概率最高词
        ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)  # 拼接到 ys
    print("Sample src:", sample_src.cpu().numpy())  # 打印源序列
    print("Sample tgt:", sample_tgt.cpu().numpy())  # 打印目标序列
    print("Model pred:", ys[:, 1:].cpu().numpy())  # 打印模型预测（去掉 BOS）

# 运行 demo 训练（在沙箱中演示几轮）
demo_train()  # 开始示例训练与演示操作
