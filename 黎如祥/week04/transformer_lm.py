# -*- coding: utf-8 -*-
"""
transformer_lm.py —— 字符级语言模型（Transformer 版，第四周作业）

任务：给一段字符序列，预测下一个字符；把"整段话"的概率分解成一串"下一个字"的概率。
      训练目标仍是交叉熵，评估指标仍是困惑度 PPL —— 和课堂的 RNN / LSTM 版完全相同，
      换掉的只有"怎么读这段序列"：RNN 一步步串行地读，Transformer 一次并行地看完。

用法：
    python transformer_lm.py                                   # 默认参数（Pre-LN + 可学习位置编码 + 权重共享）
    python transformer_lm.py --pos sinusoidal --no-norm-first   # 换成原版论文的写法
    python transformer_lm.py --corpus corpus_general.txt        # 换成全量语料（CPU 上一轮约 4 分钟）

【和课堂 language_model.py（RNN/LSTM 版）的差异】
  1. 模型：Embedding → RNN/LSTM → Linear
           改成 Embedding(+位置编码) → TransformerEncoder × N → LayerNorm → Linear
  2. 新增因果掩码（causal mask）：语言模型只能看"左边"，不能让第 i 个位置偷看第 i+1 个字符，
     否则训练时答案全在眼前，loss 会异常低、生成时却完全不会写
  3. 数据管道、损失函数、PPL 计算、最优模型保存逻辑全部沿用课堂那份，一行没改

【为什么 RNN 版不用掩码、Transformer 版必须用】
  RNN 是串行的：算第 i 步的隐藏状态时，模型物理上只见过前 i 个字符，天生"看不见未来"。
  Transformer 是并行的：一次把整句 64 个位置一起算，注意力矩阵里每个位置都能直接读到所有位置 ——
  包括后面的。所以必须手工把"右上角"（未来位置）填成 -inf，softmax 之后这些位置的权重恰好为 0。
"""

import os

# 解决 OpenMP 多线程冲突，规避 Anaconda 环境下 matplotlib + torch 一起用时的 "OMP: Error #15" 报错
# 必须在 import torch / matplotlib 之前设置
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import math
import glob
import time
import random
import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 绘图要在无显示器的环境下也能跑（比如后台任务、服务器）
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 脚本所在目录：语料、模型、曲线默认都放这儿，这样从任何目录启动都不会找错文件
HERE = os.path.dirname(os.path.abspath(__file__))


def _resolve(path):
    """相对路径优先按"当前目录"找，找不到就按"脚本所在目录"找（默认文件都在脚本旁边）。"""
    if os.path.isabs(path) or os.path.exists(path):
        return path
    return os.path.join(HERE, path)


# ─────────────────────────── 数据（与课堂版一致）───────────────────────────

def load_corpus(pattern="*.txt"):
    """
    把匹配到的所有 txt 拼成一大段文本。

    :param pattern: 文件通配符；默认只读一个文件，传 "*.txt" 就是读当前目录全部 txt
    :return: 拼好的长字符串
    """
    texts = []
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8", errors="ignore") as f:
            one = f.read()
        print(f"  读入 {os.path.basename(path)}：{len(one):,} 字符")
        texts.append(one)
    return "".join(texts)


def build_vocab(text, max_vocab=None):
    """
    构建字表：按出现频率取前 max_vocab 个字（None = 全部字符）。

    语料里的生僻字超出字表后会被 CharDataset 过滤掉 —— 字表越小模型越小、训练越快，
    但覆盖不足会丢信息。这是词表大小的取舍，作业里沿用课堂的 4000。

    :param text: 全部语料
    :param max_vocab: 字表上限；0 或 None 表示不限
    :return: (char2idx, idx2char)，字符 ↔ 整数下标两张表
    """
    from collections import Counter
    counter = Counter(text)
    chars = [c for c, _ in counter.most_common(max_vocab)] if max_vocab else sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


class CharDataset(Dataset):
    """
    字符级语言模型的训练样本：输入 x = 连续 seq_len 个字符，目标 y = 往后错一位。

    stride 控制切块间隔：
        stride=1        每个字符位置都当一个样本（相邻样本几乎完全重叠，样本数 = 语料字符数）
        stride=seq_len  不重叠切块（标准做法，样本数 = 语料长度 / seq_len）
    课堂实测用的是 stride=64，本作业沿用 —— 否则样本数被放大 64 倍，一轮训练要跑几小时。

    :param text: 纯文本（训练段或验证段）
    :param char2idx: 字符 → 下标
    :param seq_len: 每个样本的字符数
    :param stride: 相邻样本的起点间隔
    """

    def __init__(self, text, char2idx, seq_len, stride=1):
        self.seq_len = seq_len
        self.stride = max(1, stride)
        # 字表外的字符直接丢掉（它们没有下标，喂不进 Embedding）
        ids = [char2idx[c] for c in text if c in char2idx]
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return max(0, (len(self.data) - self.seq_len - 1) // self.stride + 1)

    def __getitem__(self, idx):
        start = idx * self.stride
        x = self.data[start: start + self.seq_len]            # 当前 64 个字
        y = self.data[start + 1: start + self.seq_len + 1]    # 往后错一位 = 这 64 个字各自的"下一个字"
        return x, y


# ─────────────────────────── 模型 ───────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    """
    原版 Transformer 的 sin/cos 固定位置编码（Attention is all you need, 3.5 节）。

    公式（偶数维用 sin、奇数维用 cos，频率按 10000^(2i/d) 递减）：
        PE[pos, 2i]   = sin(pos / 10000^(2i/d_model))
        PE[pos, 2i+1] = cos(pos / 10000^(2i/d_model))

    为什么需要它：自注意力是"集合运算"——把「猫追狗」换成「狗追猫」，注意力分数只由两两向量决定，
    模型自己不知道谁在前。位置编码把"第几个字"这条信息直接加进输入向量，顺序才有了意义。
    这里用固定公式而不是学习出来的参数，好处是长度外推（训练没见过更长的句子，公式照样算得出来）。
    """

    def __init__(self, d_model, max_len=4096):
        super().__init__()
        # pe: [max_len, d_model]，每一行就是一个位置的位置向量
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)          # [max_len, 1]
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                        * (-math.log(10000.0) / d_model))                          # [d_model/2]
        pe[:, 0::2] = torch.sin(pos * div)    # 偶数列
        pe[:, 1::2] = torch.cos(pos * div)    # 奇数列
        # register_buffer：跟着 model.to(device) 一起搬，但不算可训练参数、不写进 state_dict 之外的地方
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)              # [1, max_len, d_model]

    def forward(self, x):
        """:param x: [B, T, d_model] 的词向量；返回同样 shape 的"词向量 + 位置向量" """
        return x + self.pe[:, :x.size(1)]


class TransformerLM(nn.Module):
    """
    字符级语言模型：Embedding → 位置编码 → TransformerEncoder × N（带因果掩码）→ LayerNorm → Linear

    shape 变化（B=batch，T=seq_len=64，d=d_model=128，V=词表 4000）：
        x                 [B, T]          整数下标
        embed(x)          [B, T, d]       每个字查表得到 128 维向量
        + 位置编码        [B, T, d]
        encoder(...)      [B, T, d]       每层做"多头自注意力 + 前馈网络"，每个位置都与其它位置交换信息
        norm(h)           [B, T, d]
        fc(...)           [B, T, V]       每个位置输出"下一个字"在 4000 个字上的得分 logits

    两个作业里最容易被忽略、又必须做对的点：

    一、因果掩码（causal mask）
        注意力矩阵是 T×T：第 i 行 = 位置 i 去"看"所有位置。语言模型的目标是预测下一个字，
        位置 i 最多只能看到 0..i。掩码把右上三角（j > i）填成 -inf，softmax 后这些位置权重为 0。

    二、Pre-LN 还是 Post-LN
        BERT 是 Post-LN：LayerNorm(x + Sublayer(x))，层数一深就不太好训。
        GPT-2 之后普遍用 Pre-LN：x + Sublayer(LayerNorm(x))，残差主路不被归一化，梯度更好走。
        本作业用 Pre-LN（norm_first=True），并在最后补一个 LayerNorm —— 这是 GPT 的标准做法：
        Pre-LN 的每层输出都是"没归一化的残差和"，最后一层没有 LN 包着，得自己补一个。

    :param vocab_size: 词表大小 V
    :param d_model: 隐藏维度 d（= 词向量维度）
    :param nhead: 注意力头数（d 必须能被它整除）
    :param num_layers: Transformer Block 层数
    :param dim_ff: 前馈网络中间层维度（原版是 4d，小模型可以小一点）
    :param max_len: 支持的最大序列长度（位置编码 / 掩码缓存的大小）
    :param dropout: 各处的 dropout 概率
    :param pos: "learned"（BERT 式可学习位置向量）/ "sinusoidal"（原版 sin/cos）
    :param norm_first: True = Pre-LN（推荐），False = Post-LN（BERT 原版）
    :param tie_weights: 输出层与 Embedding 共享权重（语言模型的标准省参数做法）
    """

    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, dim_ff=256,
                 max_len=512, dropout=0.3, pos="learned", norm_first=True, tie_weights=True):
        super().__init__()
        assert d_model % nhead == 0, f"d_model({d_model}) 必须能被 nhead({nhead}) 整除"
        self.d_model = d_model
        self.max_len = max_len
        self.tie_weights = tie_weights

        self.embed = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.use_learned_pos = (pos == "learned")
        if self.use_learned_pos:
            # BERT 的做法：位置向量也是一张可学习的表，第 i 个位置查第 i 行
            # 缺点：只认识 0..max_len-1，比它长的句子没地方查
            self.pos_emb = nn.Embedding(max_len, d_model)
        else:
            self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",     # BERT/Transformer 常用 GELU，比 ReLU 平滑
            batch_first=True,      # 输入输出都是 [B, T, d]
            norm_first=norm_first,
        )
        # enable_nested_tensor=False：关掉 PyTorch 的推理加速路径，避免传自定义 mask 时的兼容性告警
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)

        # Pre-LN 的最后一道归一化（GPT 里叫 ln_f）
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size, bias=False)

        # 因果掩码提前算好存成 buffer：右上三角（j > i，即"未来"）填 -inf
        causal = torch.triu(torch.full((max_len, max_len), float("-inf")), diagonal=1)
        self.register_buffer("causal_mask", causal, persistent=False)

        self.apply(self._init_weights)   # 标准初始化：权重取 N(0, 0.02)，偏置为 0

        if tie_weights:
            # 输出层和词向量共享同一个权重矩阵：
            # 参数量从 2×V×d 降到 V×d，而且"认得某个字"和"输出某个字"用的是同一套表示
            self.fc.weight = self.embed.weight

    @staticmethod
    def _init_weights(module):
        """Transformer 的标准初始化。全 0 或默认均匀分布在小模型上容易出现梯度问题。"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        """:param x: [B, T] 整数下标；:return: [B, T, V] 每个位置对"下一个字"的 logits"""
        B, T = x.shape
        assert T <= self.max_len, f"序列长度 {T} 超过 max_len {self.max_len}"

        h = self.embed(x)                                  # [B, T] → [B, T, d]
        if self.use_learned_pos:
            pos_ids = torch.arange(T, device=x.device)      # [T]
            h = h + self.pos_emb(pos_ids)                   # [B, T, d] + [T, d]
        else:
            # 原版论文的做法：词向量先乘 sqrt(d) 放大，再和 sin/cos 位置编码相加
            h = self.pos_enc(h * math.sqrt(self.d_model))

        h = self.drop(h)
        # mask 是 float 的 -inf 矩阵：[T, T]，右上三角为 -inf
        # 送进注意力后：score = QKᵀ/√d + mask → 未来位置的 score 变成 -inf → softmax 权重为 0
        h = self.encoder(h, mask=self.causal_mask[:T, :T])
        return self.fc(self.norm(h))                        # [B, T, d] → [B, T, V]


# ─────────────────────────── 训练 / 评估（与课堂版一致）───────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train=True, clip=0.0):
    """
    跑一整轮，返回 (平均交叉熵, 困惑度 PPL)。

    PPL = exp(平均交叉熵)。直觉：PPL=50 表示模型在预测下一个字时，平均像在 50 个字里犹豫。
    随机猜的 PPL 就是词表大小（4000），所以 PPL 从 4000 一路降到几十，就是模型在"学会说话"。

    交叉熵要按 token 数加权平均，不能按 batch 平均：
    最后一个 batch 可能不满，按 batch 平均会让它的权重被放大。

    :param loader: 数据加载器
    :param train: True 训练（反传更新），False 只评估
    :param clip: 梯度裁剪阈值，>0 时启用；Transformer 偶尔会梯度爆掉，裁剪到 1.0 很常见
    """
    model.train(train)
    total_loss = 0.0
    total_tokens = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)                # [B, T, V]
        # logits 从 [B,T,V] 拉平成 [B*T, V]，标签拉平成 [B*T]，交叉熵对每个字符位置算一次
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        if train:
            optimizer.zero_grad()
            loss.backward()
            if clip and clip > 0:
                # 把所有参数梯度的整体范数限制在 clip 以内（超了按比例缩）
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

        total_loss += loss.item() * y.numel()   # 乘上 token 数，最后才能算"每个 token 的平均 loss"
        total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    return avg_loss, math.exp(avg_loss)


# ─────────────────────────── 生成（语言模型的"考试"）───────────────────────────

@torch.no_grad()
def generate(model, char2idx, idx2char, prompt, max_new_chars=80, temperature=1.0, device="cpu"):
    """
    自回归生成：喂一段开头，一个字一个字往后写。

    每一步做三件事：
      1. 把"最近 max_len 个字"送进模型（位置编码和掩码都只准备了这么长）
      2. 取最后一个位置的输出 → softmax 得到"下一个字"的概率分布
      3. 按概率采样一个字（temperature 越大越发散、越小越保守），拼到序列末尾，循环

    为什么只留最近若干字：Transformer 的位置编码/掩码缓存有长度上限，
    生成长文本时必须滑窗（RNN 版天然只记 hidden state，没有这个问题）。

    :param prompt: 开头几个字（必须在字表里，否则会被跳过）
    :param temperature: 采样温度；1.0 原样采样，0.5 更稳，1.5 更疯
    """
    model.eval()
    ids = [char2idx[c] for c in prompt if c in char2idx]
    if not ids:
        return prompt + "（提示词里的字都不在字表里）"

    x = torch.tensor([ids], dtype=torch.long, device=device)
    out_chars = list(prompt)

    for _ in range(max_new_chars):
        logits = model(x[:, -model.max_len:])[0, -1]      # 只看最后一个位置 → [V]
        probs = torch.softmax(logits / temperature, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        out_chars.append(idx2char[next_id])
        x = torch.cat([x, torch.tensor([[next_id]], device=device)], dim=1)

    return "".join(out_chars)


@torch.no_grad()
def show_next_chars(model, char2idx, idx2char, contexts, topk=5, device="cpu"):
    """
    给出几个上下文，打印模型认为最可能的接下来 topk 个字及其概率。

    比"自由生成一大段"更能看清模型学到了什么：
    生成是一路采样、错一个字后面就全跑偏；而"下一个字的前 5 名"是模型最直接的判断，
    上下文很短也能看出它有没有学会词语搭配。

    :param contexts: 若干个上下文句子（不在字表里的字会被跳过）
    """
    model.eval()
    for ctx in contexts:
        ids = [char2idx[c] for c in ctx if c in char2idx]
        if not ids:
            continue
        x = torch.tensor([ids[-model.max_len:]], dtype=torch.long, device=device)
        probs = torch.softmax(model(x)[0, -1], dim=-1)     # 最后一个位置的输出 → 下一个字的分布
        top = probs.topk(topk)
        pairs = "  ".join(f"{idx2char[int(i)]}({float(p):.2f})"
                          for p, i in zip(top.values, top.indices))
        print(f"  「{ctx}」→  {pairs}")


# ─────────────────────────── 主函数 ───────────────────────────

def main():
    parser = argparse.ArgumentParser(description="字符级语言模型（Transformer 版）")
    # 数据
    parser.add_argument("--corpus",     default="corpus_100k.txt", help="语料文件；可写成 '*.txt' 读全部")
    parser.add_argument("--seq_len",    type=int,   default=64)
    parser.add_argument("--stride",     type=int,   default=64, help="切块间隔：1=密集重叠（极慢），seq_len=不重叠")
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--max_vocab",  type=int,   default=4000, help="字表上限，0=不限")
    parser.add_argument("--val_ratio",  type=float, default=0.05)
    # 模型
    parser.add_argument("--d_model",    type=int,   default=128, help="隐藏维度（替代 RNN 版的 embed_dim）")
    parser.add_argument("--nhead",      type=int,   default=4, help="注意力头数，d_model 必须能被整除")
    parser.add_argument("--num_layers", type=int,   default=2, help="Transformer Block 层数")
    parser.add_argument("--dim_ff",     type=int,   default=256, help="前馈网络中间维度")
    parser.add_argument("--pos",        default="learned", choices=["learned", "sinusoidal"],
                        help="位置编码：learned=BERT 式可学习向量，sinusoidal=原版 sin/cos")
    parser.add_argument("--no-norm-first", dest="norm_first", action="store_false",
                        help="改成 Post-LN（BERT 原版写法），默认是 Pre-LN")
    parser.add_argument("--no-tie",     dest="tie_weights", action="store_false",
                        help="不共享输出层与词向量权重（默认共享）")
    parser.add_argument("--dropout",    type=float, default=0.3)
    # 训练
    parser.add_argument("--epochs",     type=int,   default=12, help="CPU 上一轮约 54 秒，12 轮约 11 分钟")
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW 的权重衰减")
    parser.add_argument("--clip",       type=float, default=1.0, help="梯度裁剪阈值，0=不裁剪")
    parser.add_argument("--seed",       type=int,   default=42)
    # 输出
    parser.add_argument("--save",       default="best_model_transformer.pt")
    parser.add_argument("--plot",       default="lm_curve.png", help="训练曲线图片路径，空字符串=不画")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    args.corpus = _resolve(args.corpus)
    args.save = args.save if os.path.isabs(args.save) else os.path.join(HERE, args.save)
    if args.plot:
        args.plot = args.plot if os.path.isabs(args.plot) else os.path.join(HERE, args.plot)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  model: TRANSFORMER  pos: {args.pos}  "
          f"norm: {'Pre-LN' if args.norm_first else 'Post-LN'}")

    # ---------- 数据 ----------
    text = load_corpus(args.corpus)
    if not text:
        raise FileNotFoundError(f"未找到语料文件：{args.corpus}，请确认路径正确。")
    print(f"语料字符数: {len(text):,}")

    char2idx, idx2char = build_vocab(text, args.max_vocab or None)
    vocab_size = len(char2idx)
    print(f"词表大小: {vocab_size}")

    # 按行打乱再切分：同一篇/同一话题的句子不能全落在训练集里，否则验证集 PPL 会虚低
    lines = text.splitlines()
    random.shuffle(lines)
    split = int(len(lines) * (1 - args.val_ratio))
    train_text = "\n".join(lines[:split])
    val_text = "\n".join(lines[split:])

    train_ds = CharDataset(train_text, char2idx, args.seq_len, args.stride)
    val_ds = CharDataset(val_text, char2idx, args.seq_len, args.stride)
    print(f"训练样本: {len(train_ds):,}   验证样本: {len(val_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # ---------- 模型 ----------
    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_ff=args.dim_ff,
        max_len=args.seq_len,      # 位置编码只需要支持到 seq_len
        dropout=args.dropout,
        pos=args.pos,
        norm_first=args.norm_first,
        tie_weights=args.tie_weights,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}"
          f"（其中词向量 {vocab_size * args.d_model:,}，权重共享={args.tie_weights}）")

    criterion = nn.CrossEntropyLoss()
    # Transformer 一般配 AdamW：它把"权重衰减"从梯度里解耦出来，正则更干净
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_ppl = float("inf")
    history = []      # [(epoch, train_loss, train_ppl, val_loss, val_ppl), ...]
    t0 = time.time()

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  "
          f"{'Val Loss':>10}  {'Val PPL':>9}  {'用时':>7}")
    print("-" * 68)

    for epoch in range(1, args.epochs + 1):
        te = time.time()
        tr_loss, tr_ppl = run_epoch(model, train_loader, criterion, optimizer, device,
                                    train=True, clip=args.clip)
        with torch.no_grad():
            va_loss, va_ppl = run_epoch(model, val_loader, criterion, optimizer, device,
                                        train=False)

        history.append((epoch, tr_loss, tr_ppl, va_loss, va_ppl))

        # 只保存验证 PPL 创新低的那一版：训练轮数给多了也不怕后期过拟合
        marker = ""
        if va_ppl < best_val_ppl:
            best_val_ppl = va_ppl
            marker = "  *"
            torch.save({
                "model_state": model.state_dict(),
                "char2idx": char2idx,
                "idx2char": idx2char,
                "args": vars(args),
            }, args.save)

        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  "
              f"{va_loss:>10.4f}  {va_ppl:>9.2f}  {time.time() - te:>6.1f}s{marker}")

    print(f"\n训练完成，总耗时 {time.time() - t0:.0f} 秒。最佳验证 PPL: {best_val_ppl:.2f}")

    # ---------- 曲线 ----------
    if args.plot:
        epochs = [h[0] for h in history]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(epochs, [h[2] for h in history], marker="o", label="train")
        ax[0].plot(epochs, [h[4] for h in history], marker="o", label="val")
        ax[0].set_title("PPL (lower is better)")
        ax[0].set_xlabel("epoch")
        ax[0].set_ylabel("perplexity")
        ax[0].legend()
        ax[0].grid(alpha=0.3)
        ax[1].plot(epochs, [h[1] for h in history], marker="o", color="tab:red", label="train")
        ax[1].plot(epochs, [h[3] for h in history], marker="o", color="tab:orange", label="val")
        ax[1].set_title("cross entropy")
        ax[1].set_xlabel("epoch")
        ax[1].set_ylabel("loss")
        ax[1].legend()
        ax[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"训练曲线已保存：{args.plot}")

    # ---------- 用最优模型现场写几段，看它学了什么 ----------
    print("\n--- 生成示例（用验证 PPL 最低那一版模型）---")
    ckpt = torch.load(args.save, map_location=device, weights_only=False)
    best = TransformerLM(
        vocab_size=len(ckpt["char2idx"]),
        d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers,
        dim_ff=args.dim_ff, max_len=args.seq_len, dropout=args.dropout,
        pos=args.pos, norm_first=args.norm_first, tie_weights=args.tie_weights,
    ).to(device)
    best.load_state_dict(ckpt["model_state"])

    for prompt in ["今天天气", "人工智能是", "我想去吃"]:
        print(f"  「{prompt}」→ {generate(best, ckpt['char2idx'], ckpt['idx2char'], prompt, 60, 1.0, device)}")

    print("\n--- 下一个字的前 5 名候选（比自由生成更能看出学到什么）---")
    show_next_chars(best, ckpt["char2idx"], ckpt["idx2char"],
                    ["今天天气", "我想去", "人工智能", "我们明天", "这部电影"],
                    topk=5, device=device)


if __name__ == "__main__":
    main()
