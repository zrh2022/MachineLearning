import math

import numpy

from tokenizer import Tokenizer
import torch
from positional_encoding import PositionalEncoding
from multi_head_attention import MultiHeadAttention
from layer_norm import LayerNorm
from feed_layer import PositionwiseFeedForward

# 测试文本
texts = [
    "hello world this is a test",
    "world peace and love",
    "hello there how are you",
    "this is another test sentence",
    "machine learning is fascinating",
    "the sun is very shine and the water flows cross the mountain"
]

seq_max_tokens = 10  # 源输入序列的最大长度
heads = 8  # 注意力头数
d_model = 768  # 这是GPT2预训练的每个词元对应的词向量的维度
d_k = d_v = 96  # d_k, d_v等于 d_model // heads
dropout = 0.1
d_ff = 512  # 全连接层的隐藏神经元数量

# 将句子转到成token_ids
tokenizer = Tokenizer()
tokens_list = [tokenizer.get_tokens(text) for text in texts]  # shape：[6, ?]
print(tokens_list)
token_ids_list = [tokenizer.get_token_ids(text) for text in texts]  # shape：[6, ?]
print(token_ids_list)

# 2. Padding
padded_token_ids_list = []
for ids in token_ids_list:
    if len(ids) > seq_max_tokens:  # 如果超过了最多的词元数目
        padded = ids[:seq_max_tokens]  # 截断
    else:
        padded = ids + [0] * (seq_max_tokens - len(ids))  # 填充
    padded_token_ids_list.append(padded)
print(padded_token_ids_list)  # shape：[6, 10]

# shape：[6, 10, 748],但是实际为列表，每个列表里有一个[10, 768]的tensor
token_vectors_list = [tokenizer.get_vectors_by_indices(ids) for ids in padded_token_ids_list]
print(token_vectors_list)

# 使用stack，会在指定维度堆叠
token_tensor = torch.stack(token_vectors_list, dim=0) * math.sqrt(d_model)
# 实际的输入向量
print("转到到tensor：" + str(token_tensor.shape))  # shape：[6, 10, 768]

pe = PositionalEncoding(d_model=d_model, max_len=seq_max_tokens)
token_tensor_add_pe = pe.forward(token_tensor)
print("添加位置编码后：" + str(token_tensor_add_pe.shape))

multi_head_attention = MultiHeadAttention(d_model=d_model, n_heads=heads, dropout=dropout)
mha_tensor = multi_head_attention.forward(token_tensor_add_pe, token_tensor_add_pe, token_tensor_add_pe)
print("通过注意力层" + str(mha_tensor.shape))

layer_norm1 = LayerNorm(d_model=d_model)
ln_tensor1 = layer_norm1.forward(mha_tensor)
print("通过归一化层1：" + str(ln_tensor1.shape))

fd_layer = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
fn_tensor = fd_layer.forward(ln_tensor1)
print("通过前馈层" + str(fn_tensor.shape))

layer_norm2 = LayerNorm(d_model=d_model)
ln_tensor2 = layer_norm2.forward(fn_tensor)
print("通过归一化层2：" + str(ln_tensor2.shape))
