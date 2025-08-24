# transformer_gpu.py
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# -------- 配置设备（优先使用 GPU） --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备：", device)
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

# -------- 超参数（可按需修改） --------
vocab_size = 10000      # 词表大小（示例用）
d_model = 512           # 嵌入维度 / model维度
nhead = 8               # multi-head 注意力头数
num_encoder_layers = 3
num_decoder_layers = 3
dim_feedforward = 2048
dropout = 0.1
max_seq_len = 50
batch_size = 32
lr = 1e-4
epochs = 5

# -------- 位置编码 --------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return x

# -------- Transformer模型封装（Encoder-Decoder） --------
class TransformerSeq2Seq(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8,
                 num_encoder_layers=3, num_decoder_layers=3,
                 dim_feedforward=2048, dropout=0.1, max_seq_len=512):
        super().__init__()
        self.d_model = d_model
        # token embedding
        self.src_tok_emb = nn.Embedding(vocab_size, d_model)
        self.tgt_tok_emb = nn.Embedding(vocab_size, d_model)
        # positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)
        self.pos_decoder = PositionalEncoding(d_model, max_len=max_seq_len)
        # core transformer
        self.transformer = nn.Transformer(d_model=d_model,
                                          nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout,
                                          batch_first=True)  # batch_first True -> (B, S, E)
        # output projection
        self.generator = nn.Linear(d_model, vocab_size)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None,
                src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # src/tgt: (B, S)
        src_emb = self.src_tok_emb(src) * math.sqrt(self.d_model)  # (B, S, d_model)
        tgt_emb = self.tgt_tok_emb(tgt) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)
        tgt_emb = self.pos_decoder(tgt_emb)

        output = self.transformer(src_emb, tgt_emb,
                                  src_mask=src_mask,
                                  tgt_mask=tgt_mask,
                                  src_key_padding_mask=src_key_padding_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask)
        # output: (B, T, d_model)
        return self.generator(output)  # (B, T, vocab_size)

# -------- 掩码函数：生成下三角掩码，防止看到未来信息 --------
def generate_square_subsequent_mask(sz):
    # returns (sz, sz) mask with float('-inf') for masked positions, 0 for allowed.
    mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
    mask = mask.to(device)
    return mask  # this is bool mask; PyTorch Transformer accepts bool masks in recent versions

# -------- 示例训练（使用随机数据演示） --------
def example_train():
    # 随机数据：src和tgt都是整数序列
    # 真实任务请替换为你的 Dataset / DataLoader
    num_samples = 1000
    src_seqs = torch.randint(1, vocab_size, (num_samples, max_seq_len), dtype=torch.long)
    tgt_seqs = torch.randint(1, vocab_size, (num_samples, max_seq_len), dtype=torch.long)

    dataset = TensorDataset(src_seqs, tgt_seqs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TransformerSeq2Seq(vocab_size=vocab_size, d_model=d_model, nhead=nhead,
                               num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
                               dim_feedforward=dim_feedforward, dropout=dropout, max_seq_len=max_seq_len)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 假设0为padding
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        t0 = time.time()
        for batch_idx, (src_batch, tgt_batch) in enumerate(loader):
            # 假设 tgt 输入是整个目标序列，训练时通常给 decoder 输入 <sos> + targets[:-1]
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)
            tgt_input = tgt_batch[:, :-1]     # decoder 输入
            tgt_output = tgt_batch[:, 1:]     # 预测目标
            # masks
            tgt_mask = generate_square_subsequent_mask(tgt_input.size(1))  # (T, T)
            # 若有 padding 则要构造 key_padding_mask： (B, S) bool where True means pad
            src_key_padding_mask = (src_batch == 0)  # 假设 pad index = 0
            tgt_key_padding_mask = (tgt_input == 0)

            optimizer.zero_grad()
            logits = model(src_batch, tgt_input,
                           tgt_mask=tgt_mask,
                           src_key_padding_mask=src_key_padding_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=src_key_padding_mask)  # (B, T, V)
            # reshape for loss: (B*T, V) 和 (B*T)
            logits_flat = logits.reshape(-1, logits.size(-1))
            tgt_flat = tgt_output.reshape(-1)
            loss = criterion(logits_flat, tgt_flat)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        t1 = time.time()
        print(f"Epoch {epoch} | loss={epoch_loss/len(loader):.4f} | time={t1-t0:.1f}s")

    # 保存模型
    torch.save(model.state_dict(), "transformer_example.pth")
    print("训练完成，模型已保存： transformer_example.pth")
    return model

# -------- 推理函数（贪心解码示例） --------
@torch.no_grad()
def greedy_decode(model, src, max_len, start_symbol=1):
    # src: (1, S) 单条样本
    model.eval()
    src = src.to(device)
    src_key_padding_mask = (src == 0)
    # 初始 decoder 输入只有 start token
    ys = torch.ones(1, 1, dtype=torch.long).to(device) * start_symbol
    for i in range(max_len - 1):
        tgt_mask = generate_square_subsequent_mask(ys.size(1))
        out = model(src, ys, tgt_mask=tgt_mask,
                    src_key_padding_mask=src_key_padding_mask,
                    tgt_key_padding_mask=(ys==0),
                    memory_key_padding_mask=src_key_padding_mask)  # (1, T, V)
        prob = out[:, -1, :]  # (1, V)
        next_word = torch.argmax(prob, dim=-1).unsqueeze(1)  # (1,1)
        ys = torch.cat([ys, next_word], dim=1)
        # 可选：遇到 EOS token（比如2）就 break
        # if next_word.item() == eos_index:
        #     break
    return ys  # (1, T)

# -------- 主入口 --------
if __name__ == "__main__":
    # 训练一个小模型（随机数据示例）
    model = example_train()

    # 推理示例
    # 随机生成一个 src，实际使用时替换为真实输入并做 tokenization
    src_example = torch.randint(1, vocab_size, (1, max_seq_len), dtype=torch.long)
    decoded = greedy_decode(model, src_example, max_len=30, start_symbol=1)
    print("解码结果（token ids）:", decoded.tolist())
