# -*- coding: utf-8 -*-  # 指定文件编码为UTF-8，支持中文注释
# 复现论文：Denoising Diffusion Probabilistic Models (DDPM, Ho et al., 2020) 的核心网络与训练/推理流程  # 顶部说明本脚本目的
# 要求满足：Python + PyTorch + GPU加速 + 含训练与推理模块 + 每行非空代码均有注释  # 概述实现要点

import os  # 导入os以便处理路径与环境变量
import math  # 导入math用于数学运算（如正弦余弦位置编码）
import argparse  # 导入argparse用于命令行参数解析
from dataclasses import dataclass  # 导入dataclass用于配置的结构化定义
from typing import Optional, Tuple  # 导入类型注解提高可读性
import torch  # 导入PyTorch主库
import torch.nn as nn  # 导入神经网络模块
import torch.nn.functional as F  # 导入常用函数式API
import torchvision  # 导入torchvision用于数据集与图像处理
from torchvision import transforms  # 导入图像预处理
from torch.utils.data import DataLoader  # 导入DataLoader用于批处理数据
from PIL import Image  # 导入PIL以便保存生成样本
from datetime import datetime  # 导入datetime用于时间戳命名
import itertools  # 导入itertools用于一些迭代工具（如分批保存图像）


# ==========================
#          配置体
# ==========================

@dataclass  # 使用dataclass定义一个简单的配置类
class Config:  # 定义配置类用于集中管理超参数
    image_size: int = 32  # 图像尺寸（CIFAR-10为32x32）
    channels: int = 3  # 图像通道数（RGB为3）
    base_channels: int = 128  # UNet基础通道数（网络宽度）
    channel_mults: Tuple[int, ...] = (1, 2, 2, 2)  # 各层倍增系数，决定下采样每级的通道数
    num_res_blocks: int = 2  # 每个尺度的残差块数量
    dropout: float = 0.1  # 残差块内的dropout比例
    use_attention_resolutions: Tuple[int, ...] = (16,)  # 在哪些分辨率上使用自注意力（以分辨率边长表示）
    num_heads: int = 4  # 自注意力头数
    timesteps: int = 1000  # DDPM总时间步数T
    beta_schedule: str = "linear"  # beta调度方案（此处实现linear与cosine两种）
    learning_rate: float = 2e-4  # Adam优化器学习率
    weight_decay: float = 0.0  # 权重衰减系数
    batch_size: int = 128  # 训练批大小
    epochs: int = 50  # 训练轮数
    ema_decay: float = 0.9999  # EMA衰减率，用于稳定采样
    grad_clip: float = 1.0  # 梯度裁剪阈值
    sample_steps: Optional[int] = None  # 采样时的步数（None表示用全T步DDPM）
    sample_eta: float = 0.0  # DDIM风格的stochasticity参数（0表示确定性DDIM）
    data_root: str = "./data"  # 数据集根目录
    out_dir: str = "./runs/ddpm"  # 输出目录（权重与样本）
    seed: int = 42  # 随机种子以保证可复现
    num_workers: int = 4  # DataLoader的并行加载线程数
    log_interval: int = 100  # 日志打印间隔（以迭代步为单位）


# ==========================
#       工具与初始化
# ==========================

def set_seed(seed: int) -> None:  # 定义设置随机种子的函数
    torch.manual_seed(seed)  # 设置CPU随机种子
    torch.cuda.manual_seed_all(seed)  # 设置所有GPU随机种子
    import random  # 导入random用于Python内建随机
    random.seed(seed)  # 设置Python随机种子
    import numpy as np  # 导入numpy用于数组相关随机
    np.random.seed(seed)  # 设置numpy随机种子


def exists(x):  # 定义一个辅助函数判断对象是否存在
    return x is not None  # 返回是否非None


def default(val, d):  # 定义一个辅助函数提供默认值
    return val if exists(val) else d  # 如果val存在返回val，否则返回默认d


# ==========================
#       时间步嵌入模块
# ==========================

class SinusoidalPosEmb(nn.Module):  # 定义正弦余弦时间步位置编码模块
    def __init__(self, dim: int):  # 构造函数，dim为输出维度
        super().__init__()  # 调用父类构造
        self.dim = dim  # 保存维度到成员变量

    def forward(self, t: torch.Tensor) -> torch.Tensor:  # 前向函数，输入时间步张量t [B]
        device = t.device  # 获取设备信息以保证计算在同一设备
        half_dim = self.dim // 2  # 取一半维度用于sin/cos配对
        emb = math.log(10000) / (half_dim - 1)  # 按Transformer定义计算频率分布
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)  # 生成指数衰减频率向量
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)  # 计算t乘以频率，形状[B, half_dim]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # 拼接sin和cos得到[B, dim]
        if self.dim % 2 == 1:  # 如果维度是奇数
            emb = F.pad(emb, (0, 1))  # 使用零填充到奇数维度
        return emb  # 返回时间嵌入


# ==========================
#        基础积木块
# ==========================

def conv3x3(in_ch, out_ch):  # 定义3x3卷积的简写构造函数
    return nn.Conv2d(in_ch, out_ch, 3, padding=1)  # 返回一个带padding的3x3卷积


class ResidualBlock(nn.Module):  # 残差块：保证GroupNorm的组数不大于通道数，避免报错
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, dropout: float, groups: int = 32):  # 构造函数
        super().__init__()  # 调用父类构造函数
        g1 = min(groups, in_ch)  # 将GroupNorm组数限制为不超过输入通道数
        self.norm1 = nn.GroupNorm(g1, in_ch)  # 第一层GroupNorm，num_channels必须等于in_ch
        self.act1 = nn.SiLU()  # 激活函数SiLU
        self.conv1 = conv3x3(in_ch, out_ch)  # 第一层3x3卷积，将in_ch映射到out_ch
        self.time_proj = nn.Linear(time_emb_dim, out_ch)  # 时间嵌入投影到out_ch维度
        g2 = min(groups, out_ch)  # 限制第二个GroupNorm的组数不超过out_ch
        self.norm2 = nn.GroupNorm(g2, out_ch)  # 第二层GroupNorm
        self.act2 = nn.SiLU()  # 第二个激活函数
        self.dropout = nn.Dropout(dropout)  # dropout层
        self.conv2 = conv3x3(out_ch, out_ch)  # 第二个3x3卷积，通道保持out_ch
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()  # 若in/out通道不匹配则用1x1做捷径投影

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:  # 前向函数
        h = self.norm1(x)  # 归一化1
        h = self.act1(h)  # 激活1
        h = self.conv1(h)  # 卷积1
        h = h + self.time_proj(t_emb)[:, :, None, None]  # 添加时间嵌入（广播到H,W）
        h = self.norm2(h)  # 归一化2
        h = self.act2(h)  # 激活2
        h = self.dropout(h)  # dropout
        h = self.conv2(h)  # 卷积2
        return h + self.skip(x)  # 返回残差连接结果



class AttentionBlock(nn.Module):  # 定义自注意力块（用于特定分辨率）
    def __init__(self, channels: int, num_heads: int):  # 构造函数，指定通道数与头数
        super().__init__()  # 调用父类构造
        self.norm = nn.GroupNorm(32, channels)  # 使用GroupNorm进行归一化
        self.qkv = nn.Conv2d(channels, channels * 3, 1)  # 使用1x1卷积同时生成Q,K,V
        self.proj = nn.Conv2d(channels, channels, 1)  # 1x1卷积用于输出投影
        self.num_heads = num_heads  # 保存头数
        self.scale = (channels // num_heads) ** -0.5  # 计算缩放因子保证数值稳定

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 前向函数
        b, c, h, w = x.shape  # 读取输入的批、通道、高、宽
        y = self.norm(x)  # 先做归一化
        qkv = self.qkv(y)  # 通过1x1卷积得到qkv张量
        q, k, v = qkv.chunk(3, dim=1)  # 按通道维拆分成Q,K,V
        q = q.reshape(b, self.num_heads, c // self.num_heads, h * w)  # 重排Q为多头形状
        k = k.reshape(b, self.num_heads, c // self.num_heads, h * w)  # 重排K
        v = v.reshape(b, self.num_heads, c // self.num_heads, h * w)  # 重排V
        attn = torch.einsum('b h d n, b h d m -> b h n m', q * self.scale, k)  # 计算注意力分数QK^T
        attn = attn.softmax(dim=-1)  # 对最后一维做softmax得到注意力权重
        out = torch.einsum('b h n m, b h d m -> b h d n', attn, v)  # 根据注意力权重与V计算输出
        out = out.reshape(b, c, h, w)  # 恢复回[B, C, H, W]形状
        out = self.proj(out)  # 通过投影卷积整合
        return out + x  # 残差连接返回


class Downsample(nn.Module):  # 定义下采样模块
    def __init__(self, channels: int):  # 构造函数
        super().__init__()  # 调用父类构造
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)  # 使用步长2的卷积进行下采样

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 前向函数
        return self.conv(x)  # 返回下采样结果


class Upsample(nn.Module):  # 定义上采样模块
    def __init__(self, channels: int):  # 构造函数
        super().__init__()  # 调用父类构造
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)  # 上采样后使用3x3卷积细化特征

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 前向函数
        x = F.interpolate(x, scale_factor=2, mode='nearest')  # 使用最近邻插值扩大尺寸
        return self.conv(x)  # 卷积后输出


# ==========================
#           UNet
# ==========================

class UNet(nn.Module):  # UNet主干用于预测噪声ε_theta
    def __init__(self, config: Config):  # 构造函数接收配置
        super().__init__()  # 父类构造
        self.config = config  # 保存配置引用
        ch = config.base_channels  # 基础通道数
        time_dim = ch * 4  # 时间嵌入维度设置为4*ch
        self.time_mlp = nn.Sequential(  # 时间嵌入的MLP
            SinusoidalPosEmb(time_dim),  # 正弦余弦嵌入
            nn.Linear(time_dim, time_dim),  # 线性层
            nn.SiLU(),  # 激活
            nn.Linear(time_dim, time_dim),  # 线性层
        )  # 结束时间MLP
        self.in_conv = nn.Conv2d(config.channels, ch, 3, padding=1)  # 输入卷积将RGB映射到ch

        self.downs = nn.ModuleList()  # 下采样模块的列表
        self.down_channels = []  # 记录每个保存点的通道数，便于后续解码对齐
        in_ch = ch  # 当前通道数初始为ch
        reso = config.image_size  # 当前分辨率
        attn_res = set(config.use_attention_resolutions)  # 需要注意力的分辨率集合
        for i, mult in enumerate(config.channel_mults):  # 遍历每个尺度的倍增因子
            out_ch = ch * mult  # 该尺度的输出通道
            for _ in range(config.num_res_blocks):  # 在该尺度堆叠若干残差块
                self.downs.append(ResidualBlock(in_ch, out_ch, time_dim, config.dropout))  # 添加残差块 (in_ch -> out_ch)
                self.down_channels.append(out_ch)  # 记录该点的输出通道数（保存点）
                in_ch = out_ch  # 更新当前通道
                if reso in attn_res:  # 若此分辨率需要注意力
                    self.downs.append(AttentionBlock(in_ch, config.num_heads))  # 添加注意力块
                    self.down_channels.append(in_ch)  # 记录注意力输出通道数
            if i != len(config.channel_mults) - 1:  # 若不是最后一个尺度则下采样
                self.downs.append(Downsample(in_ch))  # 添加下采样模块
                self.down_channels.append(in_ch)  # 记录下采样后的通道（对齐点）
                reso //= 2  # 分辨率减半

        self.mid_block1 = ResidualBlock(in_ch, in_ch, time_dim, config.dropout)  # 瓶颈残差块 1
        self.mid_attn = AttentionBlock(in_ch, config.num_heads)  # 瓶颈注意力
        self.mid_block2 = ResidualBlock(in_ch, in_ch, time_dim, config.dropout)  # 瓶颈残差块 2

        self.ups = nn.ModuleList()  # 上采样模块列表
        saved_channels = list(reversed(self.down_channels))  # 反转保存点通道列表以便pop对齐
        for i, mult in reversed(list(enumerate(config.channel_mults))):  # 逆序遍历尺度
            out_ch = ch * mult  # 目标输出通道数
            for _ in range(config.num_res_blocks):  # 上采样阶段每尺度循环
                skip_ch = saved_channels.pop() if saved_channels else out_ch  # 从保存点取skip的通道数（若为空则用out_ch）
                in_ch_for_block = in_ch + skip_ch  # 拼接后传入残差块的输入通道数
                self.ups.append(ResidualBlock(in_ch_for_block, out_ch, time_dim, config.dropout))  # 残差块接受拼接后的通道
                in_ch = out_ch  # 更新当前通道数为out_ch
                if reso in attn_res:  # 若此分辨率需要注意力
                    self.ups.append(AttentionBlock(in_ch, config.num_heads))  # 添加注意力模块
            if i != 0:  # 若不是最高分辨率则添加上采样模块
                self.ups.append(Upsample(in_ch))  # 添加上采样
                reso *= 2  # 分辨率加倍

        self.out_norm = nn.GroupNorm(32, in_ch)  # 输出前GroupNorm，num_channels等于当前in_ch
        self.out_act = nn.SiLU()  # 输出激活
        self.out_conv = nn.Conv2d(in_ch, config.channels, 3, padding=1)  # 最终输出卷积映射回图像通道数

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:  # 前向函数
        t_emb = self.time_mlp(t)  # 生成时间嵌入
        hs = []  # 保存下采样阶段的中间特征以便skip连接
        h = self.in_conv(x)  # 输入卷积
        hs.append(h)  # 保存初始特征为第一个skip点
        for m in self.downs:  # 迭代下采样模块
            if isinstance(m, ResidualBlock):  # 如果是残差块
                h = m(h, t_emb)  # 前向并传入时间嵌入
                hs.append(h)  # 保存输出用于解码阶段拼接
            else:  # 否则为Attention或Downsample
                h = m(h)  # 直接调用前向
                hs.append(h)  # 保存输出
        h = self.mid_block1(h, t_emb)  # 瓶颈残差块1
        h = self.mid_attn(h)  # 瓶颈注意力
        h = self.mid_block2(h, t_emb)  # 瓶颈残差块2
        for m in self.ups:  # 迭代上采样模块
            if isinstance(m, ResidualBlock):  # 若模块为残差块则需要拼接skip
                skip = hs.pop()  # 弹出对应的skip特征
                h = torch.cat([h, skip], dim=1)  # 在通道维拼接
                h = m(h, t_emb)  # 将拼接后的张量传入残差块
            else:  # Attention或Upsample模块
                h = m(h)  # 直接调用模块前向
        h = self.out_conv(self.out_act(self.out_norm(h)))  # 输出归一化->激活->卷积
        return h  # 返回预测噪声ε



# ==========================
#            DDPM
# ==========================

class DDPM:  # 定义DDPM对象封装噪声调度、前向扩散与反向采样
    def __init__(self, config: Config, device: torch.device):  # 构造函数接收配置与设备
        self.config = config  # 保存配置
        self.device = device  # 保存设备
        betas = self.make_beta_schedule(config.beta_schedule, config.timesteps)  # 生成beta序列
        self.betas = betas.to(device)  # 将beta放到设备上
        self.alphas = 1.0 - self.betas  # 计算alpha=1-beta
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)  # 计算累积ᾱ_t
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]],
                                             dim=0)  # 计算ᾱ_{t-1}
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)  # 预计算√ᾱ_t
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)  # 预计算√(1-ᾱ_t)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (
                    1.0 - self.alphas_cumprod)  # 计算后验方差Σ_t
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))  # 后验方差的log并裁剪避免数值问题
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (
                    1.0 - self.alphas_cumprod)  # 计算后验均值系数1
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (
                    1.0 - self.alphas_cumprod)  # 计算后验均值系数2

    @staticmethod  # 声明为静态方法不依赖实例状态
    def make_beta_schedule(schedule: str, timesteps: int) -> torch.Tensor:  # 生成beta序列
        if schedule == "linear":  # 如果选择线性调度
            beta_start = 1e-4  # 线性起始值
            beta_end = 0.02  # 线性终止值
            return torch.linspace(beta_start, beta_end, timesteps)  # 返回线性从小到大的beta
        elif schedule == "cosine":  # 如果选择余弦调度（参考Improved DDPM）
            s = 0.008  # 偏移常数稳定起点
            steps = timesteps + 1  # 计算steps便于差分
            x = torch.linspace(0, timesteps, steps)  # 从0到T均匀划分
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi / 2) ** 2  # 计算余弦ᾱ_t
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # 归一化使得ᾱ_0=1
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])  # 根据ᾱ差分得到beta
            return betas.clamp(1e-8, 0.999)  # 裁剪避免极端值
        else:  # 其他未知调度抛错
            raise ValueError(f"unknown beta schedule: {schedule}")  # 抛出异常提示

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: Optional[torch.Tensor] = None) -> torch.Tensor:  # 前向扩散：采样x_t
        if noise is None:  # 如果未提供噪声
            noise = torch.randn_like(x0)  # 生成与x0同形状的标准正态噪声
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)  # 取对应t的√ᾱ_t并reshape
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)  # 取√(1-ᾱ_t)并reshape
        return sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise  # 根据闭式公式生成x_t

    def predict_start_from_noise(self, xt: torch.Tensor, t: torch.Tensor,
                                 noise: torch.Tensor) -> torch.Tensor:  # 根据x_t与预测噪声反推x0
        sqrt_recip_alphas_cumprod_t = (1.0 / self.sqrt_alphas_cumprod[t]).view(-1, 1, 1, 1)  # 计算1/√ᾱ_t
        sqrt_recipm1_alphas_cumprod_t = (1.0 / self.sqrt_alphas_cumprod[t] - 1).view(-1, 1, 1, 1)  # 计算1/√ᾱ_t - 1
        return sqrt_recip_alphas_cumprod_t * xt - sqrt_recipm1_alphas_cumprod_t * noise  # 使用逆变换公式得到x0估计

    def p_mean_variance(self, model: nn.Module, xt: torch.Tensor, t: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:  # 计算p_theta的均值与方差
        eps_theta = model(xt, t)  # 用UNet预测噪声ε
        x0_pred = self.predict_start_from_noise(xt, t, eps_theta)  # 根据预测噪声反推x0
        x0_pred = x0_pred.clamp(-1.0, 1.0)  # 将x0裁剪到[-1,1]防止越界
        mean = (self.posterior_mean_coef1[t].view(-1, 1, 1, 1) * x0_pred + self.posterior_mean_coef2[t].view(-1, 1, 1,
                                                                                                             1) * xt)  # 计算后验均值
        var = self.posterior_variance[t].view(-1, 1, 1, 1)  # 取对应t的后验方差
        log_var = self.posterior_log_variance_clipped[t].view(-1, 1, 1, 1)  # 取对应t的对数方差
        return mean, var, log_var  # 返回均值、方差与log方差

    @torch.no_grad()  # 在采样阶段不需要梯度
    def p_sample(self, model: nn.Module, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:  # 单步反向采样x_{t-1}
        mean, var, log_var = self.p_mean_variance(model, xt, t)  # 计算p_theta的均值与方差
        noise = torch.randn_like(xt)  # 采样标准正态噪声
        nonzero_mask = (t != 0).float().view(-1, 1, 1, 1)  # 当t>0时添加噪声，t=0时不加噪声
        return mean + nonzero_mask * torch.exp(0.5 * log_var) * noise  # 生成x_{t-1}

    @torch.no_grad()  # 不需要梯度
    def sample(self, model: nn.Module, shape: Tuple[int, int, int, int]) -> torch.Tensor:  # 使用全DDPM步数生成样本
        b, c, h, w = shape  # 解析形状
        xt = torch.randn(shape, device=self.device)  # 从纯高斯噪声开始
        for t in reversed(range(self.config.timesteps)):  # 从T-1到0逐步去噪
            t_batch = torch.full((b,), t, device=self.device, dtype=torch.long)  # 构造批量时间步张量
            xt = self.p_sample(model, xt, t_batch)  # 执行单步采样
        return xt  # 返回最终生成的x0近似


# ==========================
#         训练与评估
# ==========================

class EMA:  # 定义指数滑动平均（EMA）用于更稳定的推理
    def __init__(self, model: nn.Module, decay: float):  # 构造函数
        self.decay = decay  # 保存衰减率
        self.shadow = {}  # 用字典保存影子权重
        self.device = next(model.parameters()).device  # 记录设备
        for name, param in model.named_parameters():  # 遍历模型参数
            if param.requires_grad:  # 只对需要梯度的参数
                self.shadow[name] = param.data.clone()  # 保存初始拷贝

    def update(self, model: nn.Module):  # 训练过程中更新EMA
        for name, param in model.named_parameters():  # 遍历参数
            if param.requires_grad:  # 只更新可训练参数
                assert name in self.shadow  # 确保字典中存在该键
                new_avg = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data  # 计算EMA平均
                self.shadow[name] = new_avg.clone()  # 保存更新后的影子权重

    def copy_to(self, model: nn.Module):  # 将EMA权重复制到模型中（常用于评估）
        for name, param in model.named_parameters():  # 遍历参数
            if name in self.shadow:  # 如果影子中有该参数
                param.data.copy_(self.shadow[name])  # 用EMA权重覆盖


def get_dataloader(config: Config) -> DataLoader:  # 定义获取CIFAR-10数据集的函数
    transform = transforms.Compose([  # 定义数据预处理与增强
        transforms.RandomHorizontalFlip(),  # 随机水平翻转增强鲁棒性
        transforms.ToTensor(),  # 转为张量
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 归一化到[-1,1]
    ])  # 结束Compose
    dataset = torchvision.datasets.CIFAR10(root=config.data_root, train=True, download=True,
                                           transform=transform)  # 加载CIFAR-10训练集
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers,
                        pin_memory=True)  # 构建DataLoader
    return loader  # 返回数据加载器


def save_image_grid(tensor: torch.Tensor, path: str, nrow: int = 8) -> None:  # 保存生成图像为网格
    tensor = tensor.clamp(-1, 1)  # 先裁剪到[-1,1]
    tensor = (tensor + 1) / 2  # 映射到[0,1]
    grid = torchvision.utils.make_grid(tensor, nrow=nrow)  # 生成网格图像
    ndarr = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')  # 转为numpy的HWC并缩放到[0,255]
    Image.fromarray(ndarr).save(path)  # 保存为png文件


def train(config: Config) -> None:  # 定义训练流程
    set_seed(config.seed)  # 设置随机种子
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 选择设备（优先GPU）
    if device.type != 'cuda':  # 如果没有CUDA可用
        print('[警告] 未检测到GPU，将在CPU上运行（可运行但很慢）')  # 打印提示
    os.makedirs(config.out_dir, exist_ok=True)  # 创建输出目录
    loader = get_dataloader(config)  # 获取数据加载器
    model = UNet(config).to(device)  # 初始化UNet并移动到设备
    ddpm = DDPM(config, device)  # 初始化DDPM调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)  # 定义AdamW优化器
    ema = EMA(model, config.ema_decay)  # 初始化EMA跟踪器
    global_step = 0  # 初始化全局步数计数
    for epoch in range(config.epochs):  # 遍历训练轮数
        for i, (x, _) in enumerate(loader):  # 遍历每个批次数据
            model.train()  # 切换到训练模式
            x = x.to(device, non_blocking=True)  # 将输入图像移动到设备
            b = x.size(0)  # 读取批大小
            t = torch.randint(0, config.timesteps, (b,), device=device).long()  # 在[0,T)中随机选择时间步
            noise = torch.randn_like(x)  # 生成与x同形状的标准正态噪声
            xt = ddpm.q_sample(x, t, noise)  # 前向扩散得到x_t
            pred_noise = model(xt, t)  # UNet预测噪声ε
            loss = F.mse_loss(pred_noise, noise)  # 使用MSE损失拟合ε（论文目标函数）
            optimizer.zero_grad(set_to_none=True)  # 清零梯度以节省内存
            loss.backward()  # 反向传播计算梯度
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)  # 进行梯度裁剪以稳定训练
            optimizer.step()  # 优化器更新参数
            ema.update(model)  # 使用当前参数更新EMA影子权重
            global_step += 1  # 全局步数自增
            if global_step % config.log_interval == 0:  # 到达日志间隔时
                print(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] epoch={epoch} step={global_step} loss={loss.item():.4f}")  # 打印训练日志
        # 每个epoch结束后进行一次EMA模型采样与权重保存
        ema_model = UNet(config).to(device)  # 新建一个同结构模型用于载入EMA权重
        ema.copy_to(ema_model)  # 将EMA权重复制到该模型
        with torch.no_grad():  # 采样无需梯度
            samples = ddpm.sample(ema_model, (64, config.channels, config.image_size, config.image_size))  # 生成64张样本
        save_path = os.path.join(config.out_dir, f"samples_e{epoch:03d}.png")  # 拼接样本保存路径
        save_image_grid(samples, save_path, nrow=8)  # 保存采样图像到文件
        torch.save({'model': model.state_dict(), 'ema': ema.shadow, 'config': config.__dict__},
                   os.path.join(config.out_dir, f"ckpt_e{epoch:03d}.pt"))  # 保存检查点


def infer(config: Config, ckpt_path: str, num_images: int = 64) -> None:  # 定义推理流程从权重生成图像
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 选择设备
    if device.type != 'cuda':  # 若无GPU则提示
        print('[警告] 未检测到GPU，将在CPU上运行（可运行但很慢）')  # 打印提示
    model = UNet(config).to(device)  # 构建UNet并移动到设备
    ddpm = DDPM(config, device)  # 构建DDPM调度器
    ckpt = torch.load(ckpt_path, map_location=device)  # 加载检查点
    if 'ema' in ckpt and ckpt['ema'] is not None:  # 如果包含EMA权重
        model.load_state_dict(ckpt['model'])  # 先加载普通权重（为了严格键匹配）
        ema = EMA(model, config.ema_decay)  # 构造EMA对象以便使用其copy逻辑
        ema.shadow = ckpt['ema']  # 将保存的EMA影子赋值回去
        ema.copy_to(model)  # 覆盖到模型权重
    else:  # 否则直接加载模型权重
        model.load_state_dict(ckpt['model'])  # 加载模型参数
    model.eval()  # 切换到评估模式
    with torch.no_grad():  # 推理不求梯度
        batches = (num_images + 63) // 64  # 计算需要多少批次（每批64张）
        all_imgs = []  # 用于收集所有样本
        for _ in range(batches):  # 逐批生成
            imgs = ddpm.sample(model,
                               (min(64, num_images), config.channels, config.image_size, config.image_size))  # 采样一批
            all_imgs.append(imgs)  # 收集样本
        imgs = torch.cat(all_imgs, dim=0)[:num_images]  # 合并并截断到需求数量
    os.makedirs(config.out_dir, exist_ok=True)  # 确保输出目录存在
    save_path = os.path.join(config.out_dir,
                             f"sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{num_images}.png")  # 生成保存路径
    save_image_grid(imgs, save_path, nrow=int(math.sqrt(min(64, num_images)) + 0.5))  # 保存图像网格
    print(f"已保存生成样本到: {save_path}")  # 打印保存信息


# ==========================
#            主函数
# ==========================

def build_config_from_args(args: argparse.Namespace) -> Config:  # 从命令行参数构建配置
    cfg = Config()  # 先使用默认配置
    for k, v in vars(args).items():  # 遍历命令行参数键值
        if v is not None and hasattr(cfg, k):  # 若该参数在配置中存在
            setattr(cfg, k, v)  # 用命令行值覆盖默认值
    return cfg  # 返回配置对象


def main():  # 定义脚本入口
    parser = argparse.ArgumentParser(description="DDPM(2020) 复现：UNet + 训练/推理")  # 创建参数解析器
    parser.add_argument("--mode", type=str, choices=["train", "infer"], required=True,
                        help="运行模式：train 或 infer")  # 选择运行模式
    parser.add_argument("--ckpt", type=str, default=None, help="推理模式下权重文件路径")  # 指定权重路径
    parser.add_argument("--image_size", type=int, default=None, help="图像尺寸（默认32）")  # 可选覆盖图像尺寸
    parser.add_argument("--batch_size", type=int, default=None, help="训练批大小")  # 可选覆盖批大小
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数")  # 可选覆盖轮数
    parser.add_argument("--learning_rate", type=float, default=None, help="学习率")  # 可选覆盖学习率
    parser.add_argument("--beta_schedule", type=str, default=None, choices=["linear", "cosine"],
                        help="beta调度方案")  # 选择beta调度
    parser.add_argument("--out_dir", type=str, default=None, help="输出目录")  # 可选覆盖输出目录
    parser.add_argument("--data_root", type=str, default=None, help="数据集根目录")  # 可选覆盖数据根目录
    parser.add_argument("--num_images", type=int, default=64, help="推理时生成的图像数量")  # 推理样本数量
    args = parser.parse_args()  # 解析命令行参数
    cfg = build_config_from_args(args)  # 根据参数构建配置
    if args.mode == "train":  # 如果选择训练模式
        train(cfg)  # 调用训练函数
    else:  # 否则为推理模式
        assert args.ckpt is not None, "infer模式需要提供--ckpt"  # 推理必须提供权重
        infer(cfg, args.ckpt, num_images=args.num_images)  # 调用推理函数


if __name__ == "__main__":  # 如果作为主脚本运行
    # main()  # 调用主函数
    train(config=Config())