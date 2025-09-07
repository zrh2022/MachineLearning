import math

import torch


def getStandard(max_len=1000, sd_model=512):
    # 创建位置编码矩阵: [max_len, d_model]
    pe = torch.zeros(max_len, sd_model)
    position = torch.arange(0, max_len).unsqueeze(1).float()  # [max_len, 1]

    # 计算除数项
    div_term = torch.exp(torch.arange(0, sd_model, 2).float() *
                         -(math.log(10000.0) / sd_model))  # [d_model//2]

    # 应用sin到偶数索引
    pe[:, 0::2] = torch.sin(position * div_term)  # [max_len, d_model//2]
    # 应用cos到奇数索引
    pe[:, 1::2] = torch.cos(position * div_term)  # [max_len, d_model//2]
    return pe


# 尝试计算一下位置编码
seq_len = 1000
d_model = 512
pe = torch.zeros(seq_len, d_model)

single_pe = torch.arange(0, d_model, 2)  # (d_model/2,)
single_pe = torch.exp(-math.log(10000) / d_model * single_pe)  # (d_model/2,)

position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # (seq_len, 1)
pe[:, 0::2] = torch.sin(position * single_pe) # (1000,1) * (256,) 扩展为(1000, 256) * (256) 后进行逐个元素相乘
pe[:, 1::2] = torch.cos(position * single_pe)
print(pe)
spe = getStandard()
res = pe == spe
print(spe)
print(res)
