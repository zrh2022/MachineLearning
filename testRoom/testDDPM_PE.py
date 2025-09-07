import math
import torch

# 尝试计算一下位置编码
time = torch.tensor([200, 400], dtype=torch.float)
time_len = time.size(0)  # 其实time_len就是batch数
d_model = 512
pe = torch.zeros(time_len, d_model)

half_dim = d_model // 2
single_pe = torch.arange(0, half_dim)  # (d_model/2,)
single_pe = torch.exp(-math.log(10000) / (half_dim - 1) * single_pe)  # (d_model/2,)

# (2,) * (256,) 扩展为(200, 1) * (1, 256)，再扩展到（200,256）*（200,256），最后进行逐个元素相乘
pe[:, :half_dim] = torch.sin(time[:, None].float() * single_pe[None, :])
pe[:, half_dim:] = torch.cos(time[:, None].float() * single_pe[None, :])
print(pe)
