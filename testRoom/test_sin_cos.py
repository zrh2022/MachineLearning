import torch
import math
import matplotlib.pyplot as plt

# 参数
seq_len = 100
d_model = 64

# 构造位置
positions = torch.arange(seq_len, dtype=torch.float)

# 只用 sin
pe_sin = torch.zeros(seq_len, d_model)
div_term = torch.exp(torch.arange(0, d_model, 1) * -(math.log(10000.0) / d_model))
pe_sin = torch.sin(positions.unsqueeze(1) * div_term.unsqueeze(0))  # (seq_len, d_model)

# sin/cos 交替
pe_mix = torch.zeros(seq_len, d_model)
div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
pe_mix[:, 0::2] = torch.sin(positions.unsqueeze(1) * div_term.unsqueeze(0))
pe_mix[:, 1::2] = torch.cos(positions.unsqueeze(1) * div_term.unsqueeze(0))

# 计算点积相似度（归一化）
def cosine_similarity(pe):
    # 选定第0个位置，和后续位置点积对比
    ref = pe[0] / pe[0].norm()
    sims = torch.matmul(pe, ref) / pe.norm(dim=1)
    return sims

sims_sin = cosine_similarity(pe_sin)
sims_mix = cosine_similarity(pe_mix)

# 绘制对比
plt.plot(range(seq_len), sims_sin.numpy(), label="only sin")
# plt.plot(range(seq_len), sims_mix.numpy(), label="sin+cos")
plt.xlabel("position difference")
plt.ylabel("cosine similarity with position 0")
plt.title("Position Encoding Similarity (d_model=64)")
plt.legend()
plt.grid(True)
plt.show()
