import random

import torch  # 导入PyTorch框架
import torch.nn as nn  # 导入神经网络模块
import torch.nn.functional as F  # 导入函数式接口
import torch.optim as optim  # 导入优化器
from torch.utils.data import DataLoader  # 导入数据加载器
import torchvision  # 导入计算机视觉库
import torchvision.transforms as transforms  # 导入数据变换
import numpy as np  # 导入数值计算库
import matplotlib.pyplot as plt  # 导入绘图库
from tqdm import tqdm  # 导入进度条库
import math  # 导入数学库
import os  # 导入操作系统接口
import glob  # 导入文件匹配库
import re  # 导入正则表达式库
from matplotlib import rcParams

# 指定中文字体（Windows）
rcParams['font.sans-serif'] = ['SimHei']  # 黑体
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 检查CUDA是否可用并设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")  # 打印当前使用的设备


class SinusoidalPositionEmbedding(nn.Module):
    """
    正弦位置编码模块，用于编码时间步信息
    参考Transformer中的位置编码
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim  # 嵌入维度

    def forward(self, time):
        """
        前向传播函数
        Args:
            time: 时间步张量，形状为(batch_size,)
        Returns:
            pos_emb: 位置编码，形状为(batch_size, dim)
        """
        device = time.device  # 获取输入张量的设备
        half_dim = self.dim // 2  # 计算一半的维度
        embeddings = math.log(10000) / (half_dim - 1)  # 计算嵌入基数
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)  # 计算频率
        embeddings = time[:, None] * embeddings[None, :]  # 广播相乘
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)  # 拼接sin和cos
        return embeddings  # 返回位置编码


class ResidualBlock(nn.Module):
    """
    残差块，DDPM的核心组件
    包含时间嵌入和残差连接
    """

    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)  # 时间嵌入的线性变换

        self.block1 = nn.Sequential(  # 第一个卷积块
            nn.GroupNorm(8, in_channels),  # 组归一化
            nn.SiLU(),  # SiLU激活函数
            nn.Conv2d(in_channels, out_channels, 3, padding=1),  # 3x3卷积
        )

        self.block2 = nn.Sequential(  # 第二个卷积块
            nn.GroupNorm(8, out_channels),  # 组归一化
            nn.SiLU(),  # SiLU激活函数
            nn.Dropout(dropout),  # Dropout正则化
            nn.Conv2d(out_channels, out_channels, 3, padding=1),  # 3x3卷积
        )

        # 如果输入输出通道数不同，需要1x1卷积调整维度
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)  # 1x1卷积
        else:
            self.shortcut = nn.Identity()  # 恒等映射

    def forward(self, x, time_emb):
        """
        前向传播函数
        Args:
            x: 输入特征图，形状为(batch_size, in_channels, H, W)
            time_emb: 时间嵌入，形状为(batch_size, time_emb_dim)
        Returns:
            输出特征图，形状为(batch_size, out_channels, H, W)
        """
        h = self.block1(x)  # 通过第一个块

        # 将时间嵌入加到特征图上
        time_emb = self.time_mlp(time_emb)  # 时间嵌入线性变换
        time_emb = time_emb[:, :, None, None]  # 扩展维度以匹配特征图
        h = h + time_emb  # 加入时间信息

        h = self.block2(h)  # 通过第二个块

        return h + self.shortcut(x)  # 残差连接


class AttentionBlock(nn.Module):
    """
    自注意力块，用于捕获长距离依赖
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels  # 输入通道数

        self.norm = nn.GroupNorm(8, channels)  # 组归一化
        self.q = nn.Conv2d(channels, channels, 1)  # Query卷积
        self.k = nn.Conv2d(channels, channels, 1)  # Key卷积
        self.v = nn.Conv2d(channels, channels, 1)  # Value卷积
        self.proj_out = nn.Conv2d(channels, channels, 1)  # 输出投影

    def forward(self, x):
        """
        前向传播函数
        Args:
            x: 输入特征图，形状为(batch_size, channels, H, W)
        Returns:
            输出特征图，形状为(batch_size, channels, H, W)
        """
        b, c, h, w = x.shape  # 获取输入形状

        x_norm = self.norm(x)  # 归一化
        q = self.q(x_norm)  # 计算Query
        k = self.k(x_norm)  # 计算Key
        v = self.v(x_norm)  # 计算Value

        # 重塑为序列形式进行注意力计算
        q = q.reshape(b, c, h * w).permute(0, 2, 1)  # (b, h*w, c)
        k = k.reshape(b, c, h * w)  # (b, c, h*w)
        v = v.reshape(b, c, h * w).permute(0, 2, 1)  # (b, h*w, c)

        # 计算注意力权重
        attn = torch.bmm(q, k) * (c ** -0.5)  # 缩放点积注意力  (b, h*w, h*w)
        attn = F.softmax(attn, dim=-1)  # Softmax归一化  (b, h*w, h*w)

        # 应用注意力权重
        out = torch.bmm(attn, v)  # 注意力加权  (b, h*w, c)
        out = out.permute(0, 2, 1).reshape(b, c, h, w)  # 重塑回特征图形状
        out = self.proj_out(out)  # 输出投影

        return x + out  # 残差连接


class Downsample(nn.Module):
    """下采样模块"""

    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1)  # 2x下采样卷积
        # 公式：(h+2*padding-kernel_size) // stride + 1

    def forward(self, x):
        """前向传播，将特征图尺寸减半"""
        return self.conv(x)


class Upsample(nn.Module):
    """上采样模块"""

    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)  # 3x3卷积

    def forward(self, x):
        """前向传播，将特征图尺寸加倍"""
        x = F.interpolate(x, scale_factor=2, mode='nearest')  # 最近邻插值上采样
        return self.conv(x)  # 卷积平滑


class UNet(nn.Module):
    """
    DDPM使用的UNet网络结构
    包含编码器、解码器和跳跃连接
    """

    def __init__(self, in_channels=3, out_channels=3, time_emb_dim=128,
                 base_channels=128, channel_mults=[1, 2, 2, 2],
                 num_res_blocks=2, dropout=0.1, attention_resolutions=[16]):
        super().__init__()

        self.time_emb_dim = time_emb_dim  # 时间嵌入维度
        self.num_resolutions = len(channel_mults)  # 分辨率层数

        # 时间嵌入层
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbedding(time_emb_dim // 4),  # 正弦位置编码
            nn.Linear(time_emb_dim // 4, time_emb_dim),  # 线性变换1
            nn.SiLU(),  # 激活函数
            nn.Linear(time_emb_dim, time_emb_dim),  # 线性变换2
        )

        # 输入投影
        self.input_proj = nn.Conv2d(in_channels, base_channels, 3, padding=1)  # 输入卷积

        # 编码器部分
        self.encoder_blocks = nn.ModuleList()  # 编码器块列表
        self.downsample_blocks = nn.ModuleList()  # 下采样块列表

        ch = base_channels  # 当前通道数
        # channel_mults指示了编码器有几个块，每个块的输出通道倍数（基于baseCahnnel）
        for i, mult in enumerate(channel_mults):  # 遍历每个分辨率层
            out_ch = base_channels * mult  # 输出通道数

            # 添加残差块
            for _ in range(num_res_blocks):  # 每层多个残差块
                self.encoder_blocks.append(
                    ResidualBlock(ch, out_ch, time_emb_dim, dropout)
                )
                ch = out_ch  # 更新当前通道数

                # 在指定分辨率添加注意力
                if (32 // (2 ** i)) in attention_resolutions:  # 检查是否需要注意力
                    self.encoder_blocks.append(AttentionBlock(ch))  # 添加注意力块
            # 除了最后一层，都添加下采样
            if i < len(channel_mults) - 1:  # 不是最后一层
                self.downsample_blocks.append(Downsample(ch))  # 添加下采样
            else:
                self.downsample_blocks.append(nn.Identity())  # 最后一层不下采样

        # 中间层,对应瓶颈层
        self.middle_block = nn.Sequential(
            ResidualBlock(ch, ch, time_emb_dim, dropout),  # 残差块1
            AttentionBlock(ch),  # 注意力块
            ResidualBlock(ch, ch, time_emb_dim, dropout),  # 残差块2
        )

        # 解码器部分
        self.decoder_blocks = nn.ModuleList()  # 解码器块列表
        self.upsample_blocks = nn.ModuleList()  # 上采样块列表

        for i, mult in enumerate(reversed(channel_mults)):  # 反向遍历分辨率层
            out_ch = base_channels * mult  # 输出通道数

            # 添加残差块（注意跳跃连接会增加输入通道）
            for j in range(num_res_blocks + 1):  # 比编码器多一个块
                # 第一个块需要考虑跳跃连接
                in_ch = ch + out_ch if j == 0 else out_ch  # 跳跃连接增加通道数
                self.decoder_blocks.append(
                    ResidualBlock(in_ch, out_ch, time_emb_dim, dropout)
                )
                ch = out_ch  # 更新当前通道数

                # 在指定分辨率添加注意力
                resolution_idx = len(channel_mults) - 1 - i  # 当前分辨率索引
                if (32 // (2 ** resolution_idx)) in attention_resolutions:  # 检查是否需要注意力
                    self.decoder_blocks.append(AttentionBlock(ch))  # 添加注意力块

            # 除了最后一层，都添加上采样
            if i < len(channel_mults) - 1:  # 不是最后一层
                self.upsample_blocks.append(Upsample(ch))  # 添加上采样
            else:
                self.upsample_blocks.append(nn.Identity())  # 最后一层不上采样

        # 输出层
        self.output_proj = nn.Sequential(
            nn.GroupNorm(8, base_channels),  # 组归一化
            nn.SiLU(),  # 激活函数
            nn.Conv2d(base_channels, out_channels, 3, padding=1),  # 输出卷积
        )

    def forward(self, x, time):
        """
        UNet前向传播
        Args:
            x: 输入图像，形状为(batch_size, channels, H, W)
            time: 时间步，形状为(batch_size,)
        Returns:
            预测的噪声，形状为(batch_size, channels, H, W)
        """
        # 时间嵌入
        time_emb = self.time_embedding(time)  # 计算时间嵌入

        # 输入投影
        x = self.input_proj(x)  # 输入卷积

        # 编码器前向传播，保存跳跃连接
        skip_connections = []  # 跳跃连接列表

        block_idx = 0  # 块索引
        for i in range(self.num_resolutions):  # 遍历每个分辨率层
            # 残差块
            for _ in range(2):  # 每层2个残差块
                x = self.encoder_blocks[block_idx](x, time_emb)  # 应用残差块
                block_idx += 1  # 增加块索引

                # 检查是否有注意力块
                if block_idx < len(self.encoder_blocks) and isinstance(self.encoder_blocks[block_idx], AttentionBlock):
                    x = self.encoder_blocks[block_idx](x)  # 应用注意力
                    block_idx += 1  # 增加块索引

                skip_connections.append(x)  # 保存跳跃连接

            # 下采样
            x = self.downsample_blocks[i](x)  # 应用下采样

        # 中间层
        x = self.middle_block[0](x, time_emb)  # 第一个残差块
        x = self.middle_block[1](x)  # 注意力块
        x = self.middle_block[2](x, time_emb)  # 第二个残差块

        # 解码器前向传播，使用跳跃连接
        block_idx = 0  # 重置块索引
        for i in range(self.num_resolutions):  # 遍历每个分辨率层
            # 残差块
            for j in range(3):  # 每层3个残差块
                if j == 0:  # 第一个块使用跳跃连接
                    # 找到匹配当前x空间维度的跳跃连接
                    matching_skip = None
                    skip_to_remove = None

                    for idx, skip in enumerate(reversed(skip_connections)):
                        if skip.shape[2:] == x.shape[2:]:  # 找到空间维度匹配的
                            matching_skip = skip
                            skip_to_remove = idx
                            break

                    skip_connections.pop(skip_to_remove)  # 获取跳跃连接
                    x = torch.cat([x, matching_skip], dim=1)  # 拼接跳跃连接

                x = self.decoder_blocks[block_idx](x, time_emb)  # 应用残差块
                block_idx += 1  # 增加块索引

                # 检查是否有注意力块
                if block_idx < len(self.decoder_blocks) and isinstance(self.decoder_blocks[block_idx], AttentionBlock):
                    x = self.decoder_blocks[block_idx](x)  # 应用注意力
                    block_idx += 1  # 增加块索引

            # 上采样
            x = self.upsample_blocks[i](x)  # 应用上采样

        # 输出投影
        x = self.output_proj(x)  # 输出卷积

        return x  # 返回预测噪声


class DDPMScheduler:
    """
    DDPM调度器，管理前向和反向扩散过程
    """

    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=2e-2):
        self.num_timesteps = num_timesteps  # 总时间步数

        # 生成beta序列（方差调度）
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)  # 线性调度

        # 计算alpha相关参数
        self.alphas = 1.0 - self.betas  # alpha = 1 - beta
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)  # 累积乘积
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)  # 前一步的累积乘积

        # 计算采样需要的系数
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)  # sqrt(α̅_t)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)  # sqrt(1-α̅_t)

        # 计算反向过程的系数
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)  # 1/sqrt(α_t)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)  # 后验方差

    def add_noise(self, x_0, noise, timesteps):
        """
        前向扩散过程：给原始图像添加噪声
        Args:
            x_0: 原始图像，形状为(batch_size, channels, H, W)
            noise: 随机噪声，形状为(batch_size, channels, H, W)
            timesteps: 时间步，形状为(batch_size,)
        Returns:
            添加噪声后的图像
        """
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[timesteps][:, None, None, None]  # 获取对应时间步的系数
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[timesteps][:, None, None, None]  # 获取噪声系数

        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise  # 前向扩散公式

    def sample_prev_timestep(self, x_t, noise_pred, timestep):
        """
        反向扩散一步：从x_t采样x_{t-1}
        Args:
            x_t: 当前时间步的图像
            noise_pred: 模型预测的噪声
            timestep: 当前时间步
        Returns:
            前一时间步的图像
        """
        # 获取相关系数
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[timestep]  # 1/sqrt(α_t)
        betas_t = self.betas[timestep]  # β_t
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[timestep]  # sqrt(1-α̅_t)

        # 计算均值
        mean = sqrt_recip_alphas_t * (x_t - betas_t / sqrt_one_minus_alphas_cumprod_t * noise_pred)  # 后验均值

        if timestep == 0:  # 如果是最后一步
            return mean  # 不添加噪声
        else:
            posterior_variance_t = self.posterior_variance[timestep]  # 后验方差
            noise = torch.randn_like(x_t)  # 采样噪声
            return mean + torch.sqrt(posterior_variance_t) * noise  # 添加方差噪声


class DDPM:
    """
    DDPM主类，包含训练和推理逻辑
    """

    def __init__(self, model, scheduler, device):
        self.model = model.to(device)  # 将模型移到GPU
        self.scheduler = scheduler  # 扩散调度器
        self.device = device  # 计算设备

        # 将调度器参数移到设备上
        for attr_name in ['betas', 'alphas', 'alphas_cumprod', 'alphas_cumprod_prev',
                          'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod',
                          'sqrt_recip_alphas', 'posterior_variance']:
            setattr(self.scheduler, attr_name, getattr(self.scheduler, attr_name).to(device))

    def train_step(self, x_0):
        """
        训练一步
        Args:
            x_0: 原始图像批次
        Returns:
            损失值
        """
        batch_size = x_0.shape[0]  # 获取批次大小

        # 随机采样时间步
        timesteps = torch.randint(0, self.scheduler.num_timesteps, (batch_size,), device=self.device)  # 随机时间步

        # 采样噪声
        noise = torch.randn_like(x_0)  # 标准高斯噪声

        # 前向扩散：添加噪声
        x_t = self.scheduler.add_noise(x_0, noise, timesteps)  # 添加噪声得到x_t

        # 模型预测噪声
        noise_pred = self.model(x_t, timesteps)  # UNet预测噪声

        # 计算损失（简单的L2损失）
        loss = F.mse_loss(noise_pred, noise)  # 均方误差损失

        return loss  # 返回损失

    @torch.no_grad()  # 推理时不需要梯度
    def sample(self, shape, num_steps=None):
        """
        DDPM采样过程
        Args:
            shape: 生成图像的形状 (batch_size, channels, H, W)
            num_steps: 采样步数，默认使用全部时间步
        Returns:
            生成的图像
        """
        if num_steps is None:  # 如果没指定步数
            num_steps = self.scheduler.num_timesteps  # 使用全部时间步

        # 从纯噪声开始
        x = torch.randn(shape, device=self.device)  # 初始化纯噪声

        # 反向扩散过程
        timesteps = torch.linspace(self.scheduler.num_timesteps - 1, 0, num_steps, dtype=torch.long,
                                   device=self.device)  # 时间步序列

        for i, t in enumerate(tqdm(timesteps, desc="采样中")):  # 遍历时间步
            t_batch = t.expand(shape[0])  # 扩展到批次维度

            # 模型预测噪声
            noise_pred = self.model(x, t_batch)  # UNet预测噪声

            # 反向扩散一步
            x = self.scheduler.sample_prev_timestep(x, noise_pred, t)  # 采样前一步
        import torchvision.utils

        torchvision.utils.save_image(x, "grid.png", nrow=4, normalize=True)

        return x  # 返回生成的图像


def find_latest_checkpoint(checkpoint_dir):
    """
    查找最新的检查点文件
    Args:
        checkpoint_dir: 检查点文件夹路径
    Returns:
        latest_checkpoint: 最新检查点路径，如果没找到返回None
        start_epoch: 起始epoch，如果没找到返回0
    """
    if not os.path.exists(checkpoint_dir):  # 检查目录是否存在
        return None, 0  # 如果目录不存在，返回None和0

    # 查找所有检查点文件
    checkpoint_pattern = os.path.join(checkpoint_dir, 'ddpm_epoch_*.pth')  # 检查点文件模式
    checkpoint_files = glob.glob(checkpoint_pattern)  # 查找匹配的文件

    if not checkpoint_files:  # 如果没有找到检查点文件
        return None, 0  # 返回None和0

    # 提取epoch数字并找到最大的
    epoch_numbers = []  # epoch数字列表
    for checkpoint_file in checkpoint_files:  # 遍历检查点文件
        # 从文件名中提取epoch数字
        match = re.search(r'ddpm_epoch_(\d+)\.pth', os.path.basename(checkpoint_file))  # 正则匹配
        if match:  # 如果匹配成功
            epoch_numbers.append((int(match.group(1)), checkpoint_file))  # 添加epoch数字和文件路径

    if not epoch_numbers:  # 如果没有找到有效的epoch数字
        return None, 0  # 返回None和0

    # 找到最大的epoch数字
    latest_epoch, latest_checkpoint = max(epoch_numbers, key=lambda x: x[0])  # 找到最大epoch

    return latest_checkpoint, latest_epoch  # 返回最新检查点路径和epoch


def load_checkpoint(model, optimizer, checkpoint_path):
    """
    加载检查点
    Args:
        model: DDPM模型
        optimizer: 优化器
        checkpoint_path: 检查点文件路径
    Returns:
        start_epoch: 起始epoch
        best_loss: 最佳损失
    """
    print(f"从检查点加载模型: {checkpoint_path}")  # 提示加载信息
    checkpoint = torch.load(checkpoint_path, map_location=device)  # 加载检查点

    # 判断检查点格式（支持新旧格式）
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:  # 新格式检查点
        # 完整格式的检查点
        model.model.load_state_dict(checkpoint['model_state_dict'])  # 加载模型状态
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:  # 如果有优化器状态
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # 加载优化器状态
        start_epoch = checkpoint.get('epoch', 0)  # 获取起始epoch
        best_loss = checkpoint.get('loss', float('inf'))  # 获取最佳损失
        print(f"加载检查点成功，从epoch {start_epoch + 1}开始训练，上次损失: {best_loss:.4f}")  # 提示加载成功
    else:  # 旧格式检查点（只有模型权重）
        model.model.load_state_dict(checkpoint)  # 加载模型权重
        # 从文件名提取epoch信息
        match = re.search(r'ddpm_epoch_(\d+)\.pth', os.path.basename(checkpoint_path))  # 正则匹配
        start_epoch = int(match.group(1)) if match else 0  # 提取epoch数字
        best_loss = float('inf')  # 设置默认最佳损失
        print(f"加载旧格式检查点成功，从epoch {start_epoch + 1}开始训练")  # 提示加载成功

    return start_epoch, best_loss  # 返回起始epoch和最佳损失


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir):
    """
    保存检查点
    Args:
        model: DDPM模型
        optimizer: 优化器
        epoch: 当前epoch
        loss: 当前损失
        checkpoint_dir: 检查点保存目录
    """
    os.makedirs(checkpoint_dir, exist_ok=True)  # 创建保存目录

    checkpoint = {
        'epoch': epoch,  # 当前epoch
        'model_state_dict': model.model.state_dict(),  # 模型参数
        'optimizer_state_dict': optimizer.state_dict(),  # 优化器状态
        'loss': loss,  # 当前损失
        'scheduler_config': {  # 调度器配置
            'num_timesteps': model.scheduler.num_timesteps,  # 时间步数
            'beta_start': model.scheduler.betas[0].item(),  # 起始beta
            'beta_end': model.scheduler.betas[-1].item(),  # 结束beta
        }
    }

    checkpoint_path = os.path.join(checkpoint_dir, f'ddpm_epoch_{epoch}.pth')  # 检查点文件路径
    torch.save(checkpoint, checkpoint_path)  # 保存检查点
    print(f"检查点已保存到: {checkpoint_path}")  # 提示保存成功


def train_ddpm(model, train_loader, num_epochs=100, lr=2e-4, checkpoint_dir='checkpoints'):
    """
    训练DDPM模型（支持断点续训）
    Args:
        model: DDPM模型实例
        train_loader: 训练数据加载器
        num_epochs: 训练轮数
        lr: 学习率
        checkpoint_dir: 检查点保存目录
    """
    # 设置优化器
    optimizer = optim.Adam(model.model.parameters(), lr=lr)  # Adam优化器

    # 查找并加载最新的检查点
    latest_checkpoint, start_epoch = find_latest_checkpoint(checkpoint_dir)  # 查找最新检查点
    best_loss = float('inf')  # 初始化最佳损失

    if latest_checkpoint is not None:  # 如果找到检查点
        start_epoch, best_loss = load_checkpoint(model, optimizer, latest_checkpoint)  # 加载检查点
        start_epoch += 1  # 从下一个epoch开始训练
    else:
        start_epoch = 0  # 从头开始训练
        print("未找到检查点，从头开始训练")  # 提示从头开始

    # 训练循环
    model.model.train()  # 设置为训练模式

    for epoch in range(start_epoch, num_epochs):  # 从起始epoch开始遍历
        total_loss = 0.0  # 总损失
        num_batches = 0  # 批次数

        # 使用tqdm显示进度
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")  # 进度条

        for batch_idx, (images, _) in enumerate(pbar):  # 遍历每个批次
            images = images.to(device)  # 将图像移到GPU

            # 前向传播
            loss = model.train_step(images)  # 计算损失

            # 反向传播
            optimizer.zero_grad()  # 清零梯度
            loss.backward()  # 反向传播
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1.0)  # 梯度裁剪
            optimizer.step()  # 更新参数

            # 统计损失
            total_loss += loss.item()  # 累加损失
            num_batches += 1  # 增加批次计数

            # 更新进度条
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})  # 显示当前损失

        # 打印epoch统计
        avg_loss = total_loss / num_batches  # 计算平均损失
        print(f"Epoch {epoch + 1}, 平均损失: {avg_loss:.4f}")  # 打印平均损失

        # 更新最佳损失
        if avg_loss < best_loss:  # 如果当前损失更好
            best_loss = avg_loss  # 更新最佳损失

        # 每10个epoch保存一次模型（修改保存频率）
        if (epoch + 1) % 10 == 0:  # 每10个epoch
            save_checkpoint(model, optimizer, epoch + 10, avg_loss, checkpoint_dir)  # 保存检查点

            # 可选：删除旧的检查点以节省空间（保留最近5个）
            # cleanup_old_checkpoints(checkpoint_dir, keep_recent=5)  # 清理旧检查点


def cleanup_old_checkpoints(checkpoint_dir, keep_recent=5):
    """
    清理旧的检查点文件，只保留最近的几个
    Args:
        checkpoint_dir: 检查点目录
        keep_recent: 保留最近的检查点数量
    """
    if not os.path.exists(checkpoint_dir):  # 如果目录不存在
        return  # 直接返回

    # 查找所有检查点文件
    checkpoint_pattern = os.path.join(checkpoint_dir, 'ddpm_epoch_*.pth')  # 检查点文件模式
    checkpoint_files = glob.glob(checkpoint_pattern)  # 查找匹配的文件

    if len(checkpoint_files) <= keep_recent:  # 如果文件数量不超过保留数量
        return  # 不需要清理

    # 按epoch数字排序
    epoch_files = []  # epoch文件列表
    for checkpoint_file in checkpoint_files:  # 遍历检查点文件
        match = re.search(r'ddmp_epoch_(\d+)\.pth', os.path.basename(checkpoint_file))  # 正则匹配
        if match:  # 如果匹配成功
            epoch_files.append((int(match.group(1)), checkpoint_file))  # 添加epoch和文件路径

    # 按epoch排序，删除旧的文件
    epoch_files.sort(key=lambda x: x[0])  # 按epoch排序
    files_to_delete = epoch_files[:-keep_recent]  # 需要删除的文件

    for epoch, file_path in files_to_delete:  # 遍历需要删除的文件
        try:
            os.remove(file_path)  # 删除文件
            print(f"已删除旧检查点: {file_path}")  # 提示删除成功
        except OSError as e:
            print(f"删除检查点失败: {file_path}, 错误: {e}")  # 提示删除失败


def sample_and_visualize(model, num_samples=4):
    """
    采样并可视化生成的图像
    Args:
        model: 训练好的DDPM模型
        num_samples: 采样数量
    """
    model.model.eval()  # 设置为评估模式

    # 采样
    with torch.no_grad():  # 不需要梯度
        samples = model.sample((num_samples, 3, 32, 32))  # 生成样本
        samples = (samples + 1) / 2  # 归一化到[0,1]
        samples = torch.clamp(samples, 0, 1)  # 确保在有效范围内

    # 可视化
    fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 3, 3))  # 创建子图
    if num_samples == 1:  # 处理单个样本的情况
        axes = [axes]  # 转换为列表

    for i in range(num_samples):  # 遍历每个样本
        img = samples[i].cpu().permute(1, 2, 0).numpy()  # 转换为numpy格式
        axes[i].imshow(img)  # 显示图像
        axes[i].axis('off')  # 关闭坐标轴
        axes[i].set_title(f'Sample {i + 1}')  # 设置标题

    plt.tight_layout()  # 调整布局
    plt.savefig('generated_samples.png', dpi=150, bbox_inches='tight')  # 保存图像
    plt.show()  # 显示图像


def load_and_inference(model_path, num_samples=4):
    """
    加载训练好的模型并进行推理
    Args:
        model_path: 模型文件路径
        num_samples: 生成样本数量
    """
    # 重新创建模型结构
    scheduler = DDPMScheduler(num_timesteps=1000)  # 创建调度器
    unet = UNet(
        in_channels=3,  # RGB输入通道
        out_channels=3,  # RGB输出通道
        time_emb_dim=16,  # 时间嵌入维度
        base_channels=128,  # 基础通道数
        channel_mults=[1, 2, 2, 2],  # 通道倍数
        num_res_blocks=2,  # 残差块数量
        dropout=0.1,  # Dropout概率
        attention_resolutions=[16]  # 注意力分辨率
    )  # 创建UNet

    ddpm = DDPM(unet, scheduler, device)  # 创建DDPM实例

    # 加载训练好的权重（支持新旧格式）
    checkpoint = torch.load(model_path, map_location=device)  # 加载检查点

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:  # 新格式
        ddpm.model.load_state_dict(checkpoint['model_state_dict'])  # 加载模型权重
        print(f"模型已从 {model_path} 加载 (完整格式)")  # 提示加载成功
    else:  # 旧格式
        ddpm.model.load_state_dict(checkpoint)  # 直接加载权重
        print(f"模型已从 {model_path} 加载 (权重格式)")  # 提示加载成功

    # 进行推理
    print("开始推理生成...")  # 提示开始推理
    sample_and_visualize(ddpm, num_samples=num_samples)  # 生成样本


def fast_sampling(model, shape, num_steps=50):
    """
    快速采样：使用更少的步数进行采样
    Args:
        model: DDPM模型
        shape: 生成图像形状
        num_steps: 采样步数（少于1000）
    Returns:
        生成的图像
    """
    model.model.eval()  # 设置为评估模式

    # 选择子集时间步进行快速采样
    timesteps = torch.linspace(
        model.scheduler.num_timesteps - 1,
        0,
        num_steps,
        dtype=torch.long,
        device=model.device
    )  # 等间隔选择时间步

    # 从纯噪声开始
    x = torch.randn(shape, device=model.device)  # 初始化噪声

    # 快速反向扩散
    for i, t in enumerate(tqdm(timesteps, desc="快速采样中")):  # 遍历选定的时间步
        t_batch = t.expand(shape[0])  # 扩展时间步到批次维度

        with torch.no_grad():  # 不计算梯度
            noise_pred = model.model(x, t_batch)  # 预测噪声
            x = model.scheduler.sample_prev_timestep(x, noise_pred, t)  # 反向扩散一步

    return x  # 返回生成结果


def progressive_generation_demo(model, shape=(1, 3, 32, 32)):
    """
    演示渐进式生成过程
    Args:
        model: DDPM模型
        shape: 生成图像形状
    """
    model.model.eval()  # 设置为评估模式

    # 保存生成过程中的中间结果
    intermediate_results = []  # 中间结果列表
    # save_steps = [999, 800, 600, 400, 200, 100, 50, 0]  # 保存的时间步
    save_steps = [500, 400, 300, 200, 150, 100, 50, 0]  # 保存的时间步

    # 从纯噪声开始
    x = torch.randn(shape, device=model.device)  # 初始化噪声

    # 反向扩散过程
    for t in tqdm(range(model.scheduler.num_timesteps - 1, -1, -1), desc="渐进生成中"):  # 从T-1到0
        t_batch = torch.full((shape[0],), t, device=model.device, dtype=torch.long)  # 当前时间步

        with torch.no_grad():  # 不计算梯度
            noise_pred = model.model(x, t_batch)  # 预测噪声
            x = model.scheduler.sample_prev_timestep(x, noise_pred, t)  # 反向扩散一步

        # 保存指定时间步的结果
        if t in save_steps:  # 如果是需要保存的时间步
            sample_normalized = (x + 1) / 2  # 归一化到[0,1]
            sample_normalized = torch.clamp(sample_normalized, 0, 1)  # 限制范围
            intermediate_results.append((t, sample_normalized[0].cpu()))  # 保存结果

    # 可视化渐进生成过程
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))  # 创建2x4子图
    axes = axes.flatten()  # 展平轴数组

    for i, (timestep, sample) in enumerate(intermediate_results):  # 遍历中间结果
        img = sample.permute(1, 2, 0).numpy()  # 转换格式
        axes[i].imshow(img)  # 显示图像
        axes[i].axis('off')  # 关闭坐标轴
        axes[i].set_title(f't = {timestep}')  # 设置时间步标题

    plt.suptitle('DDPM渐进式去噪过程')  # 设置总标题
    plt.tight_layout()  # 调整布局
    plt.savefig('progressive_generation.png', dpi=150, bbox_inches='tight')  # 保存图像
    plt.show()  # 显示图像


def evaluate_model(model, test_loader, num_batches=10):
    """
    评估模型性能
    Args:
        model: DDPM模型
        test_loader: 测试数据加载器
        num_batches: 评估批次数
    """
    model.model.eval()  # 设置为评估模式
    total_loss = 0.0  # 总损失
    num_samples = 0  # 样本数

    with torch.no_grad():  # 不计算梯度
        for batch_idx, (images, _) in enumerate(test_loader):  # 遍历测试批次
            if batch_idx >= num_batches:  # 限制评估批次数
                break  # 跳出循环

            images = images.to(device)  # 移到GPU
            loss = model.train_step(images)  # 计算损失（不更新参数）

            total_loss += loss.item() * images.shape[0]  # 累加损失
            num_samples += images.shape[0]  # 累加样本数

    avg_loss = total_loss / num_samples  # 计算平均损失
    print(f"测试集平均损失: {avg_loss:.4f}")  # 打印评估结果

    return avg_loss  # 返回平均损失


def save_model_checkpoint(model, optimizer, epoch, loss, filepath):
    """
    保存完整的模型检查点（兼容函数，使用新的save_checkpoint）
    Args:
        model: DDPM模型
        optimizer: 优化器
        epoch: 当前epoch
        loss: 当前损失
        filepath: 保存路径
    """
    checkpoint_dir = os.path.dirname(filepath)  # 获取目录
    save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir)  # 调用新的保存函数


def main():
    """主函数，演示完整的训练和推理流程"""

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize(32),  # 调整图像大小
        transforms.CenterCrop(32),  # 中心裁剪
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 归一化到[-1,1]
    ])

    # 加载CIFAR-10数据集
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )  # 训练数据集

    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )  # 训练数据加载器

    # 测试数据集
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )  # 测试数据集

    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )  # 测试数据加载器

    # 初始化模型组件
    scheduler = DDPMScheduler(num_timesteps=1000)  # 创建扩散调度器
    unet = UNet(
        in_channels=3,  # RGB图像输入通道
        out_channels=3,  # RGB图像输出通道
        time_emb_dim=16,  # 时间嵌入维度
        base_channels=128,  # 基础通道数
        channel_mults=[1, 2, 2, 2],  # 各层通道倍数
        num_res_blocks=2,  # 每层残差块数量
        dropout=0.1,  # Dropout概率
        attention_resolutions=[16]  # 使用注意力的分辨率
    )  # 创建UNet模型

    ddpm = DDPM(unet, scheduler, device)  # 创建DDPM实例

    print(f"模型参数数量: {sum(p.numel() for p in ddpm.model.parameters()):,}")  # 打印参数数量

    # 训练模型（支持断点续训）
    print("开始训练...")  # 提示开始训练
    train_ddpm(ddpm, train_loader, num_epochs=200000, lr=2e-4, checkpoint_dir='checkpoints')  # 训练模型

    # 评估模型
    print("评估模型...")  # 提示开始评估
    evaluate_model(ddpm, test_loader, num_batches=10)  # 评估模型性能

    # 生成样本
    print("生成样本...")  # 提示开始生成
    sample_and_visualize(ddpm, num_samples=8)  # 生成并可视化样本

    # 演示渐进生成
    print("演示渐进生成过程...")  # 提示渐进生成
    progressive_generation_demo(ddpm)  # 渐进生成演示


def demo_tensor_shapes():
    """演示张量形状流动的函数"""
    print("=== DDPM张量形状演示 ===")  # 打印标题

    # 创建示例输入
    batch_size = 4  # 批次大小
    x = torch.randn(batch_size, 3, 32, 32).to(device)  # 示例图像
    time = torch.randint(0, 1000, (batch_size,)).to(device)  # 示例时间步

    print(f"输入图像形状: {x.shape}")  # 打印输入形状
    print(f"时间步形状: {time.shape}")  # 打印时间步形状

    # 创建模型
    scheduler = DDPMScheduler()  # 创建调度器
    model = UNet().to(device)  # 创建并移动模型到GPU

    # 前向传播
    with torch.no_grad():  # 不计算梯度
        output = model(x, time)  # 模型前向传播

    print(f"模型输出形状: {output.shape}")  # 打印输出形状
    print(f"参数总数: {sum(p.numel() for p in model.parameters()):,}")  # 打印参数数量


def simple_training_demo():
    """简单的训练演示（小数据集，快速验证）"""
    print("=== 简单训练演示 ===")  # 打印标题

    # 创建简单的合成数据
    num_samples = 100  # 样本数量
    synthetic_data = torch.randn(num_samples, 3, 32, 32)  # 合成数据
    synthetic_dataset = torch.utils.data.TensorDataset(synthetic_data, torch.zeros(num_samples))  # 创建数据集
    synthetic_loader = DataLoader(synthetic_dataset, batch_size=10, shuffle=True)  # 数据加载器

    # 创建小模型进行快速演示
    scheduler = DDPMScheduler(num_timesteps=501)  # 较少时间步
    unet = UNet(
        time_emb_dim=64,  # 较小嵌入维度
        base_channels=64,  # 较少通道数
        channel_mults=[1, 2],  # 较少层数
        num_res_blocks=2,  # 较少残差块
        attention_resolutions=[]  # 不使用注意力
    )  # 创建小模型

    ddpm = DDPM(unet, scheduler, device)  # 创建DDPM实例

    # 快速训练几个epoch（支持断点续训）
    train_ddpm(ddpm, synthetic_loader, num_epochs=3, lr=1e-3, checkpoint_dir='demo_checkpoints')  # 训练

    # 生成一个样本
    sample = ddpm.sample((1, 3, 32, 32), num_steps=20)  # 快速采样
    print(f"生成样本形状: {sample.shape}")  # 打印生成样本形状

    return ddpm  # 返回训练好的模型


def comprehensive_demo():
    """综合演示函数"""
    print("=== DDPM综合演示 ===")  # 打印标题

    # 1. 张量形状演示
    demo_tensor_shapes()  # 演示张量形状
    print("\n" + "=" * 50 + "\n")  # 分隔线

    # 2. 简单训练演示
    model = simple_training_demo()  # 简单训练演示
    print("\n" + "=" * 50 + "\n")  # 分隔线

    # 3. 采样演示
    print("=== 采样演示 ===")  # 打印标题
    sample_and_visualize(model, num_samples=4)  # 采样演示

    # 4. 渐进生成演示
    print("=== 渐进生成演示 ===")  # 打印标题
    progressive_generation_demo(model)  # 渐进生成演示


# 使用示例和测试代码
if __name__ == "__main__":
    print("DDPM完整实现")  # 打印开始信息

    # 设置随机种子以确保可重复性
    torch.manual_seed(random.randint(0, 100))  # 设置PyTorch随机种子
    np.random.seed(random.randint(0, 100))  # 设置NumPy随机种子
    if torch.cuda.is_available():  # 如果CUDA可用
        torch.cuda.manual_seed(random.randint(0, 100))  # 设置CUDA随机种子

    # 选择运行模式
    mode = "train"  # 可选: "train", "inference", "demo", "comprehensive"

    if mode == "train":  # 训练模式
        print("运行完整训练模式...")  # 提示信息
        main()  # 运行主训练函数

    elif mode == "inference":  # 推理模式
        print("运行推理模式...")  # 提示信息
        # 注意：需要先有训练好的模型文件
        model_path = "checkpoints/ddpm_epoch_200.pth"  # 模型文件路径
        if os.path.exists(model_path):  # 检查文件是否存在
            load_and_inference(model_path, num_samples=8)  # 加载并推理
        else:
            print(f"模型文件 {model_path} 不存在，请先训练模型")  # 提示文件不存在

    elif mode == "demo":  # 演示模式
        print("运行快速演示模式...")  # 提示信息
        demo_tensor_shapes()  # 张量形状演示

    elif mode == "comprehensive":  # 综合演示模式
        print("运行综合演示模式...")  # 提示信息
        comprehensive_demo()  # 综合演示

    print("DDPM实现完成！")  # 提示完成


# 额外的工具函数
def visualize_noise_schedule():
    """可视化噪声调度"""
    scheduler = DDPMScheduler()  # 创建调度器

    # 绘制beta和alpha的变化
    timesteps = np.arange(scheduler.num_timesteps)  # 时间步数组

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))  # 创建2x2子图

    # 绘制beta_t
    axes[0, 0].plot(timesteps, scheduler.betas.cpu().numpy())  # 绘制beta
    axes[0, 0].set_title('Beta Schedule')  # 设置标题
    axes[0, 0].set_xlabel('Timestep')  # x轴标签
    axes[0, 0].set_ylabel('Beta')  # y轴标签

    # 绘制alpha_t
    axes[0, 1].plot(timesteps, scheduler.alphas.cpu().numpy())  # 绘制alpha
    axes[0, 1].set_title('Alpha Schedule')  # 设置标题
    axes[0, 1].set_xlabel('Timestep')  # x轴标签
    axes[0, 1].set_ylabel('Alpha')  # y轴标签

    # 绘制累积alpha
    axes[1, 0].plot(timesteps, scheduler.alphas_cumprod.cpu().numpy())  # 绘制累积alpha
    axes[1, 0].set_title('Cumulative Alpha')  # 设置标题
    axes[1, 0].set_xlabel('Timestep')  # x轴标签
    axes[1, 0].set_ylabel('Alpha_cumprod')  # y轴标签

    # 绘制信噪比
    snr = scheduler.alphas_cumprod / (1 - scheduler.alphas_cumprod)  # 计算信噪比
    axes[1, 1].plot(timesteps, snr.cpu().numpy())  # 绘制信噪比
    axes[1, 1].set_title('Signal-to-Noise Ratio')  # 设置标题
    axes[1, 1].set_xlabel('Timestep')  # x轴标签
    axes[1, 1].set_ylabel('SNR')  # y轴标签
    axes[1, 1].set_yscale('log')  # 使用对数坐标

    plt.tight_layout()  # 调整布局
    plt.savefig('noise_schedule.png', dpi=150, bbox_inches='tight')  # 保存图像
    plt.show()  # 显示图像
