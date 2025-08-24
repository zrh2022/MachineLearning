import math

import numpy

from transformer import Transformer
from tokenizer import Tokenizer
import torch
from positional_encoding import PositionalEncoding
from multi_head_attention import MultiHeadAttention
from layer_norm import LayerNorm
from feed_layer import PositionwiseFeedForward
from seq_pre_handler import TextPreHandler
import torch.nn.functional as F


def singleGen(src, tgt_input):
    tgt_mask = model.generate_square_subsequent_mask(tgt_seq_len, device)
    tgt_mask = tgt_mask.unsqueeze(0).expand(batch_size, -1, -1)
    output = model(src, tgt=tgt_input, tgt_mask=tgt_mask)
    output_seqs = torch.argmax(output, dim=-1)  # [batch_size, tgt_seq_len]
    ori_texts = tgt_text_pre_handler.texts_post_handle(output_seqs)
    print(ori_texts)

# 测试文本
texts = [
        "The cat sleeps peacefully",
        "Birds fly in the sky",
        "Rain falls gently down",
        "Children play in park",
        "Books contain many stories",
        "Music makes people happy"
    ]

test_texts = [
    "so many flowers",
    "the sun is very shine and the water flows cross the mountain",
    "the sky is very blue and large",
    "hello world",
    "the learning rate is five percent",
    "machine learning is fascinating"
]


out_texts = ["! " + text for text in texts]
test_out_texts = ["! " + text for text in test_texts]

# 模拟训练数据
batch_size = 6
src_seq_len = 10
tgt_seq_len = 15
num_batches = 500
seq_max_tokens = 50257  # 源输入序列的最大长度
heads = 8  # 注意力头数
d_model = 768  # 这是GPT2预训练的每个词元对应的词向量的维度
d_k = d_v = 96  # d_k, d_v等于 d_model // heads
dropout = 0.1
d_ff = 512  # 全连接层的隐藏神经元数量
lr = 1e-4

src_text_pre_handler = TextPreHandler(seq_max_tokens=src_seq_len, d_model=d_model)
src = src_text_pre_handler.pre_handle_texts_v2(texts)
tgt_text_pre_handler = TextPreHandler(seq_max_tokens=tgt_seq_len, d_model=d_model)
tgt = tgt_text_pre_handler.pre_handle_texts_v2(out_texts)
test_src = src_text_pre_handler.pre_handle_texts_v2(test_texts)
test_tgt = tgt_text_pre_handler.pre_handle_texts_v2(test_out_texts)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
src = src.to(device)
tgt = tgt.to(device)
test_src = test_src.to(device)
test_tgt = test_tgt.to(device)

# 创建小型模型用于演示
model = Transformer(
    src_vocab_size=seq_max_tokens,
    tgt_vocab_size=seq_max_tokens,
    d_model=d_model,
    n_heads=heads,
    n_layers=2,
    d_ff=d_ff,
    dropout=dropout
).to(device)

# 优化器 - 使用Adam优化器
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)

print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

model.train()

# 随机生成源序列和目标序列 (避免使用pad_idx=0)
# src = torch.randint(1, seq_max_tokens, (batch_size, src_seq_len)).to(device)  # [batch_size, src_seq_len]
# tgt = torch.randint(1, seq_max_tokens, (batch_size, tgt_seq_len)).to(device)  # [batch_size, tgt_seq_len]

print(f"源序列形状: {src.shape}")
print(f"目标序列形状: {tgt.shape}")

# print("语句经过前置预处理，形状：" + str(pre_handled_seqs.shape))  # shape：[6, 10, 768]
# print("语句经过前置预处理，实际值：" + str(pre_handled_seqs))

# 创建目标掩码
tgt_mask = model.generate_square_subsequent_mask(tgt_seq_len, device)
tgt_mask = tgt_mask.unsqueeze(0).expand(batch_size, -1, -1)

for batch_idx in range(num_batches):
    # 生成随机数据（实际应用中这里是真实的训练数据）
    # src = torch.randint(1, 1000, (batch_size, src_seq_len)).to(device)
    # tgt_input = torch.randint(1, 1000, (batch_size, tgt_seq_len)).to(device)
    # tgt_output = torch.randint(1, 1000, (batch_size, tgt_seq_len)).to(device)

    # 前向传播
    output = model(src, tgt, tgt_mask=tgt_mask)  # [batch_size, tgt_seq_len, vocab_size]

    # 计算损失
    loss = F.cross_entropy(
        output.reshape(-1, output.size(-1)),  # [batch_size * tgt_seq_len, vocab_size]
        tgt.reshape(-1),  # [batch_size * tgt_seq_len]
        ignore_index=0  # 忽略填充token
    )

    # 反向传播
    optimizer.zero_grad()
    loss.backward()

    # 梯度裁剪（防止梯度爆炸）
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # 更新参数
    optimizer.step()

    print(f"批次 {batch_idx + 1}/{num_batches}: 损失={loss.item():.4f}, 学习率={lr:.2e}")

    if device.type == 'cuda':
        memory_mb = torch.cuda.memory_allocated(device) / 1024 ** 2
        print(f"  GPU内存使用: {memory_mb:.1f} MB")

# 方法1：贪心解码 - 选择概率最大的token
# 前向传播
# 初始 tgt_input: 只给一个起始 token（可以是0或特殊符号）

singleGen(src, tgt)
print("-"*10)
singleGen(test_src, test_tgt)

tgt_input = torch.zeros(batch_size, 1, dtype=torch.long).to(src.device)
output = None
for i in range(tgt_seq_len):
    # tgt_mask = model.generate_square_subsequent_mask(i+1, device)
    # tgt_mask = tgt_mask.unsqueeze(0).expand(batch_size, -1, -1)
    output = model(src, tgt=tgt_input, tgt_mask=None)
    #方式2
    probs = torch.softmax(output[:, -1, :], dim=-1)
    next_token = torch.multinomial(probs, 1)
    #方式1
    # output_seqs = torch.argmax(output, dim=-1)  # [batch_size, tgt_seq_len]
    # next_token = output_seqs[:, -1].unsqueeze(1)
    generated = torch.cat([tgt_input, next_token], dim=1)
    tgt_input = generated

ori_texts = tgt_text_pre_handler.texts_post_handle(tgt_input)
print(ori_texts)


