# -*- coding: utf-8 -*-
# 解决 OpenMP 多线程冲突，规避 Anaconda 环境下的线程报错（必须在 import torch 之前设置）
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import csv
import time
import random
import argparse
import unicodedata
from collections import Counter, OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

"""
任务描述：12 类语种识别（给一句话，判断它是哪种语言）
输入：任意一句话（UTF-8 字符串）。字符级处理，不需要分词器：
      中文按字、英文按字母、印地语按字母 + 组合符号，一视同仁。
分类规则：输出 12 个语种之一，标签就是这句话的出处（Tatoeba 逐语种导出，无需人工标注）

12 个语种 = 课堂 8 个（ar en es hi ja pt ru zh）+ 作业新增 4 个（ko 韩 / th 泰 / tr 土耳其 / vi 越南）

数据：data/langid/v12/（相对本文件所在目录）
      train 9600 条 / dev 1200 条 / test 1800 条，每个语种 800 / 100 / 150，类别严格均衡
字符表：data/langid/vocab/vocab_v12.txt
      课堂只发了 v8 的字符表，v12 这张必须自己从 train.csv 统计生成（代码里 build_char_vocab() 自动做）

模型：Embedding → RNN → 掩码 MaxPool → LayerNorm → Dropout → Linear
损失：CrossEntropyLoss（内部自带 softmax，所以模型只输出 logits）
优化器：Adam(lr=1e-3)

【这份作业相对课堂 8 类版（train_langid_cls.py）改了三处】
  1. LABELS 从 8 个补到 12 个 —— 模型输出维度 = len(LABELS)，少一个语种，
     多出来的那句就没地方落，训练时直接被 raise 拦下
  2. 字符表不再读现成的 vocab_v8.txt，而是从 v12/train.csv 重新统计生成
     —— v8 那张表里没有韩文、泰文、越南语、土耳其语用的字符，
        直接拿来用的话这些字符全部落到 1 号 <UNK>，模型等于睁眼瞎
  3. 数据目录指向 v12，并加了训练曲线、逐类准确率、混淆矩阵、UNK 占比等检查

运行：
    python train_langid_cls_v12.py                 # 训练 + 评测 + 存盘 + 画曲线
    python train_langid_cls_v12.py --epochs 15     # 想多训几轮
    python train_langid_cls_v12.py --vocab data/langid/vocab/vocab_v12.txt  # 复用已生成的词表
"""

# ─── 超参数 ────────────────────────────────────────────────────────────
SEED = 42
MAX_LEN = 40          # 数据卡里句子长度窗口是 8~40 字符，统一截断/补齐到 40
EMBED_DIM = 64        # 每个字变成一个 64 维向量
HIDDEN_DIM = 64       # RNN 隐藏状态维度
DROPOUT = 0.3         # 训练时随机丢掉 30% 的神经元，防止模型死记硬背
BATCH_SIZE = 64
LR = 1e-3             # Adam 的默认学习率，这个任务基本不用调
EPOCHS = 30           # 实测 20 轮≈96.8%、30 轮≈97.3%、40 轮≈97.2%（已到平台的区间）
NUM_THREADS = 1       # 小模型单线程反而最快：多线程的调度开销比计算本身还大

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data", "langid", "v12")
VOCAB_PATH = os.path.join(HERE, "data", "langid", "vocab", "vocab_v12.txt")
CURVE_PATH = os.path.join(HERE, "train_curve.png")
MODEL_PATH = os.path.join(HERE, "langid_rnn_v12.pt")

# 作业的 12 个语种：顺序固定，模型的第 i 个输出格子就对应第 i 个语种
# 注意别按字母重排后忘了改数据，标签→下标的映射由这里的顺序唯一决定
LABELS = ["ar", "en", "es", "hi", "ja", "ko", "pt", "ru", "th", "tr", "vi", "zh"]


# ─── 1. 文本归一化（必须和建数据时一致）────────────────────────────────
def normalize(text):
    """
    NFKC 归一化 + 转小写 + 只保留「字母(L)」与「组合符号(M)」。

    为什么保留 M（组合符号）：
      印地语、泰语的元音是以组合标记的形式跟在辅音后面的（如 कि 里的 ि），
      只保留字母会把印地语/泰语砍成一堆裸辅音，元音信息全丢，模型就分不开它们了。
      泰语（th）是这次作业新加的语种，这一条对它尤其关键。

    为什么必须和建词表时用同一个 normalize：
      字符表是按 normalize 之后的字符统计的。如果训练时归一化规则变了，
      同一个字算出来的 id 就对不上词表，等于拿错索引去查表。
    """
    t = unicodedata.normalize("NFKC", text).lower()
    return "".join(c for c in t if unicodedata.category(c)[0] in ("L", "M"))


# ─── 2. 字符表：统计 / 存盘 / 读取 ─────────────────────────────────────
def build_char_vocab(sentences, min_freq=1):
    """
    从训练集统计字符表：所有出现过的字符，每个发一个整数编号。

    编号规则：
      0 号 <PAD> —— 补齐用（Embedding 里 padding_idx=0，向量恒为 0、梯度不更新）
      1 号 <UNK> —— 没见过的字符（字符级下几乎用不到，但要留这个位置）
      其余按出现次数从高到低编号，次数相同按字符本身排序（保证每次跑结果一致）

    为什么要"从训练集统计"而不是从 dev/test：
      词表只能看训练集，否则等于提前偷看考试答案（术语叫数据泄漏）。
      dev/test 里没在训练集出现过的字符，落到 <UNK> 是正常的，也是该付的代价。

    :param sentences: 训练集的所有句子（list[str]）
    :param min_freq: 出现次数低于它的字符直接丢掉（默认 1 = 全都留下）
    :return: (vocab, freq)
             vocab —— OrderedDict{字符: id}，含 <PAD>/<UNK>
             freq  —— Counter{字符: 出现次数}，写词表文件时要用
    """
    freq = Counter()
    for s in sentences:
        freq.update(normalize(s))

    # 先按次数降序、次数相同按字符升序排，再依次发号
    chars = sorted(((c, n) for c, n in freq.items() if n >= min_freq),
                   key=lambda kv: (-kv[1], kv[0]))

    vocab = OrderedDict([("<PAD>", 0), ("<UNK>", 1)])
    for ch, _ in chars:
        vocab[ch] = len(vocab)
    return vocab, freq


def save_vocab_file(vocab, freq, path):
    """
    把字符表写成和课堂 vocab_v8.txt 完全一样的格式：
        第 0~1 行是 <PAD>/<UNK>
        之后每行「字符 \\t id \\t 出现次数」
    <PAD>/<UNK> 在 Counter 里统计不到，次数记成 0，这样三列格式统一，
    load_vocab_file() 不用为特殊行写分支。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for ch, idx in vocab.items():
            f.write(f"{ch}\t{idx}\t{freq.get(ch, 0)}\n")


def load_vocab_file(path):
    """
    读已经生成好的字符表（每行「字符 \\t id \\t 出现次数」，以 # 开头的行是注释）。

    这里只把「字符 → id」读进内存，次数那列读完就丢：
    训练和推理只需要查 id，次数是给人看的（能看出哪些字符是高频字）。
    """
    vocab = OrderedDict()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            ch, idx, _freq = line.rstrip("\n").split("\t")
            vocab[ch] = int(idx)
    return vocab


def encode(text, vocab, maxlen=MAX_LEN):
    """
    一句话 → 定长整数序列（送进模型的最终形态）。

    三步：归一化 → 逐字查 id（查不到给 1 号 <UNK>）→ 截断到 maxlen 并用 0 补到 maxlen。
    补齐必须做，因为一个 batch 要拼成矩阵，而矩阵得是矩形：
    句子长短不一时，短的那句右边补 0，长的那句砍掉尾巴。

    :param text: 原始句子
    :param vocab: 字符表 {字符: id}
    :param maxlen: 统一长度（本任务 40）
    :return: list[int]，长度恒为 maxlen
    """
    ids = [vocab.get(ch, 1) for ch in normalize(text)][:maxlen]
    ids += [0] * (maxlen - len(ids))
    return ids


# ─── 3. 读数据 ─────────────────────────────────────────────────────────
def load_split(data_dir, name):
    """
    读 dev.csv / train.csv / test.csv，返回 [(text, label), ...]。

    用 csv.DictReader 按表头取列，比按列号取稳：列顺序变了也不会错位。
    """
    path = os.path.join(data_dir, f"{name}.csv")
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((r["text"], r["label"]))
    return rows


class TextDataset(Dataset):
    """
    把 (句子, 标签) 列表包装成 PyTorch 认识的"数据集"。

    一次性把全部句子 encode 成 id 序列存起来（9600 × 40 个 int，内存毫无压力），
    这样 __getitem__ 只是取下标，不用在训练循环里反复做字符串处理。

    __getitem__ 返回：
      X[i] 转成 long 张量 —— Embedding 只认整数下标
      y[i] 保持 python int —— CrossEntropyLoss 的标签就是类别下标
    """

    def __init__(self, rows, vocab, label2id):
        self.X = [encode(t, vocab) for t, _ in rows]
        self.y = [label2id[l] for _, l in rows]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return torch.tensor(self.X[i], dtype=torch.long), self.y[i]


# ─── 4. 模型 ───────────────────────────────────────────────────────────
class LangIDClassifier(nn.Module):
    """
    Embedding → RNN → 掩码 MaxPool → LayerNorm → Dropout → Linear

    shape 变化（B=batch，L=句长 40，E=embed 64，H=hidden 64，C=类别 12）：
      x                 [B, L]       整数 id
      embedding(x)      [B, L, E]    每个字变成 64 维向量
      rnn(...) → h      [B, L, H]    每一步的隐藏状态（相当于"读到这里时，这句话是什么语种"的证据）
      max(dim=1)        [B, H]       把 L 这一步压成一个向量
      fc(pooled)        [B, C]       12 个语种的原始得分 logits

    为什么 MaxPool 前要 mask：
      补齐位喂进去的是全 0 向量，但 RNN 照样会往前走，池化时就会把"读过一串 0"的状态也算进来。
      用 masked_fill 把补齐位填成 -1e9，再取 max 就等于它们不存在（-1e9 永远当不上最大值）。

    为什么输出不加 softmax：
      CrossEntropyLoss 内部自带 log_softmax + NLL，外面再手动 softmax 就等于做了两遍，
      梯度会被压小，训练变慢甚至学不动。只有推理时想看概率才手动 softmax。
    """

    def __init__(self, vocab_size, num_classes,
                 embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, dropout=DROPOUT):
        super().__init__()
        # padding_idx=0：0 号 <PAD> 的向量固定为全 0 且永远不参与梯度更新
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # batch_first=True：输入输出都是 [B, L, H]，和上面的 shape 说明一致
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        # LayerNorm 对每个样本的最后 64 维做归一化：把 RNN 输出的数值尺度拉回稳定区间
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        # 12 个语种 → 12 个输出格子
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        :param x: [B, L] 的整数 id 张量
        :return: [B, C] 的 logits（未过 softmax）
        """
        mask = (x != 0).unsqueeze(-1)               # [B, L] → [B, L, 1]，补齐位是 False
        h, _ = self.rnn(self.embedding(x))          # [B, L, E] → [B, L, H]；_ 是最后一步的隐藏状态，这里用不到
        h = h.masked_fill(~mask, -1e9)              # 补齐位填成极小值，池化时自动被排除
        pooled = h.max(dim=1)[0]                    # [B, L, H] → [B, H]，沿句长维度取最大
        pooled = self.dropout(self.norm(pooled))    # 先归一化再随机失活
        return self.fc(pooled)                      # [B, H] → [B, C]


# ─── 5. 评测 ───────────────────────────────────────────────────────────
def run_loader(model, loader):
    """跑一遍 loader，返回 (预测下标列表, 真实下标列表)。"""
    model.eval()
    preds, golds = [], []
    with torch.no_grad():   # 评测不需要梯度，关掉能省内存、也避免误更新参数
        for X, y in loader:
            preds += model(X).argmax(dim=1).tolist()   # logits 最大那个格子 = 预测的语种
            golds += y.tolist()
    return preds, golds


def macro_f1(preds, golds, num_classes):
    """
    每一类各自算 F1，再对类别取平均。

    为什么看 macro-F1 而不只看准确率：
      准确率会被大类主导。假设 12 类里有 1 类占 90%，全猜它也能有 90% 准确率，
      但另外 11 类全废。macro-F1 对每一类一视同仁，少数类学得差会直接反映出来。
      本数据集是均衡的，所以两者数值接近，但写作业养成看 F1 的习惯不吃亏。

    F1 = 2·P·R / (P+R)，P = 预测成 c 的里面有多少真的是 c，R = 真的 c 里面找回来多少。
    """
    f1s = []
    for c in range(num_classes):
        tp = sum(p == c and g == c for p, g in zip(preds, golds))   # 真 c 且预测 c
        fp = sum(p == c and g != c for p, g in zip(preds, golds))   # 预测 c 但真不是
        fn = sum(p != c and g == c for p, g in zip(preds, golds))   # 真 c 但没预测出来
        if tp + fp + fn == 0:
            continue    # 这一类在数据里既没真样本也没被预测过，跳过（不影响平均的合理性）
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s)


def confusion(preds, golds, labels):
    """
    混淆矩阵：mat[真实类别][预测类别] = 条数。

    只用错的格子就能看出"模型把谁认成了谁"，比一个总分信息量大得多。
    本任务预期：es ↔ pt（同一套拉丁字母、词形很像）是主要混淆对；
    新加的 tr / vi 也靠拉丁字母，所以要盯一下它们会不会被 es/pt 吞掉。
    """
    n = len(labels)
    mat = [[0] * n for _ in range(n)]
    for p, g in zip(preds, golds):
        mat[g][p] += 1
    return mat


def print_confusion(mat, labels):
    """打印完整混淆矩阵（12×12 一行一类，行=真实，列=预测）+ 最常混淆的组合。"""
    print("\n  混淆矩阵（行=真实，列=预测）：")
    # 表头：语种代码太宽，只印前两个字符对齐
    print("        " + " ".join(f"{lab:>4}" for lab in labels))
    for g, lab in enumerate(labels):
        line = " ".join(f"{mat[g][p]:>4}" for p in range(len(labels)))
        print(f"    {lab:>3} {line}")

    pairs = [(mat[g][p], labels[g], labels[p])
             for g in range(len(labels)) for p in range(len(labels)) if g != p]
    pairs.sort(reverse=True)
    print("\n  最常混淆的组合（真实 → 预测）：")
    for cnt, g, p in pairs[:5]:
        if cnt == 0:
            break
        print(f"    {g} → {p}: {cnt} 条")


def unk_ratio(rows, vocab):
    """
    统计某个数据集里有百分之多少的字符不在词表里（会落到 <UNK>）。

    这是"词表是否够用"的直接体检：
      训练集正常应该接近 0%；dev/test 略高一点也正常（新句子里的新字符）。
      如果 dev/test 的 UNK 占比很高，说明词表没覆盖住，模型会明显掉分。
    """
    total = hit = 0
    for text, _ in rows:
        for ch in normalize(text):
            total += 1
            if ch not in vocab:
                hit += 1
    return hit / max(1, total)


# ─── 6. 训练 ───────────────────────────────────────────────────────────
def main():
    """训练入口：读数据 → 建/读词表 → 搭模型 → 训练 → 测试集评测 → 存盘 → 画曲线"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA_DIR, help="v12 数据目录（含 train/dev/test.csv）")
    ap.add_argument("--vocab", default=VOCAB_PATH,
                    help="字符表路径；文件存在就直接读，不存在就从训练集生成并写到这个路径")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--threads", type=int, default=NUM_THREADS)
    ap.add_argument("--save", default=MODEL_PATH, help="模型存到这个路径（空字符串 = 不存）")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    random.seed(SEED)
    torch.manual_seed(SEED)     # 固定随机种子，保证每次跑初始化的权重、打乱顺序都一样，结果可复现

    # ---- 数据 ----
    train_rows = load_split(args.data_dir, "train")
    dev_rows = load_split(args.data_dir, "dev")
    test_rows = load_split(args.data_dir, "test")

    # 数据里的标签必须都能在 LABELS 里找到，否则模型输出格子不够，训练毫无意义
    seen = sorted({l for _, l in train_rows})
    extra = [l for l in seen if l not in LABELS]
    if extra:
        raise SystemExit(
            f"数据里出现了 LABELS 之外的标签：{' '.join(extra)}\n"
            f"  当前 LABELS（{len(LABELS)} 类）：{' '.join(LABELS)}\n"
            f"  模型输出维度 = len(LABELS)，多出来的语种没有对应的输出格子。"
        )
    missing = [l for l in LABELS if l not in seen]
    if missing:
        raise SystemExit(f"LABELS 里的这些语种在训练集里一条样本都没有：{' '.join(missing)}")

    # 标签 → 下标。顺序由 LABELS 决定，所以保存模型时一定要把 LABELS 一起存，
    # 否则预测时下标对不上语种，输出就成了乱码
    label2id = {l: i for i, l in enumerate(LABELS)}
    num_classes = len(LABELS)

    # ---- 字符表：没有就现生成 ----
    if os.path.exists(args.vocab):
        vocab = load_vocab_file(args.vocab)
        print(f"字符表：复用已存在的 {os.path.relpath(args.vocab, HERE)}（{len(vocab)} 个字符）")
    else:
        vocab, freq = build_char_vocab([t for t, _ in train_rows])
        save_vocab_file(vocab, freq, args.vocab)
        print(f"字符表：从 train.csv 现场生成，已写入 {os.path.relpath(args.vocab, HERE)}"
              f"（{len(vocab)} 个字符）")

    # 词表覆盖率体检：前多少个字符能覆盖 95% 的字符出现次数
    counts = []
    with open(args.vocab, encoding="utf-8") as f:
        for line in f:
            if not line.startswith("#") and line.strip():
                counts.append(int(line.rstrip("\n").split("\t")[2]))
    counts.sort(reverse=True)
    total_chars = sum(counts)
    acc_cnt, cover_n = 0, 0
    for n in counts:
        acc_cnt += n
        cover_n += 1
        if acc_cnt >= 0.95 * total_chars:
            break
    print(f"  字符出现次数共 {total_chars:,}，前 {cover_n} 个字符覆盖 95%")

    print(f"数据目录：{os.path.relpath(args.data_dir, HERE)}")
    print(f"  训练 {len(train_rows)} 条 / 验证 {len(dev_rows)} 条 / 测试 {len(test_rows)} 条")
    print(f"  类别 {num_classes} 个：{' '.join(LABELS)}")
    print(f"  随机猜的准确率基准：{1 / num_classes:.1%}")
    print(f"  dev 的 UNK 占比：{unk_ratio(dev_rows, vocab):.4%}"
          f"  /  test 的 UNK 占比：{unk_ratio(test_rows, vocab):.4%}")

    train_loader = DataLoader(TextDataset(train_rows, vocab, label2id),
                              batch_size=BATCH_SIZE, shuffle=True)   # 训练集必须打乱，否则模型会学到"数据顺序"
    dev_loader = DataLoader(TextDataset(dev_rows, vocab, label2id), batch_size=128)
    test_loader = DataLoader(TextDataset(test_rows, vocab, label2id), batch_size=128)

    # ---- 模型 ----
    model = LangIDClassifier(len(vocab), num_classes)
    print(f"  模型参数量：{sum(p.numel() for p in model.parameters()):,}\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # ---- 训练循环 ----
    log = []    # 每轮 [dev_acc, train_loss]，最后画曲线用
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()   # 打开 Dropout（训练模式）
        total = 0.0
        for X, y in train_loader:
            logits = model(X)               # 前向：logits [B, C]
            loss = criterion(logits, y)     # 交叉熵内部已做 softmax，直接喂原始 logits
            optimizer.zero_grad()           # 清空上一批的梯度（不清会累加）
            loss.backward()                 # 反向传播，算出每个参数的梯度
            optimizer.step()                # Adam 按梯度更新参数
            total += loss.item()

        preds, golds = run_loader(model, dev_loader)
        acc = sum(p == g for p, g in zip(preds, golds)) / len(golds)
        avg_loss = total / len(train_loader)
        log.append([acc, avg_loss])
        print(f"Epoch {epoch:2d}/{args.epochs}  train_loss={avg_loss:.4f}  dev_acc={acc:.4f}")
    elapsed = time.time() - t0
    print(f"\n训练耗时：{elapsed:.0f} 秒（{args.epochs} 个 epoch，{args.threads} 线程）")

    # ---- 测试集评测 ----
    preds, golds = run_loader(model, test_loader)
    acc = sum(p == g for p, g in zip(preds, golds)) / len(golds)
    f1 = macro_f1(preds, golds, num_classes)
    print(f"\n测试集：accuracy = {acc:.4f}   macro-F1 = {f1:.4f}")
    print_confusion(confusion(preds, golds, LABELS), LABELS)

    # 逐语种准确率：均衡数据下，哪一类拖后腿一眼就看出来
    print("\n  各语种准确率：")
    for c, lab in enumerate(LABELS):
        idx = [i for i, g in enumerate(golds) if g == c]
        hit = sum(preds[i] == c for i in idx)
        print(f"    {lab}: {hit}/{len(idx)} = {hit / max(1, len(idx)):.3f}")

    # ---- 推理示例：拿测试集前 6 句看看模型实际怎么答 ----
    print("\n--- 推理示例 ---")
    model.eval()
    demos = [t for t, _ in test_rows[:6]]
    with torch.no_grad():
        for text in demos:
            logits = model(torch.tensor([encode(text, vocab)]))
            prob = torch.softmax(logits, dim=1)[0]     # 推理时手动 softmax 才有概率
            pid = int(prob.argmax())
            print(f"  [{LABELS[pid]} {prob[pid]:.2f}]  {text[:50]}")

    # ---- 存盘：权重 + 词表 + 标签 + 超参，一个文件自助 ----
    if args.save:
        torch.save({"state_dict": model.state_dict(), "vocab": dict(vocab),
                    "labels": LABELS, "maxlen": MAX_LEN,
                    "embed_dim": EMBED_DIM, "hidden_dim": HIDDEN_DIM,
                    "dropout": DROPOUT}, args.save)
        print(f"\n模型已保存：{os.path.relpath(args.save, HERE)}")

    # ---- 画训练曲线 ----
    epochs = list(range(1, len(log) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [x[0] for x in log], marker="o", color="tab:blue")
    axes[0].set_title("dev accuracy")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("accuracy")
    axes[0].grid(alpha=0.3)
    axes[1].plot(epochs, [x[1] for x in log], marker="o", color="tab:red")
    axes[1].set_title("train loss")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("loss")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(CURVE_PATH, dpi=150)
    print(f"训练曲线已保存：{os.path.relpath(CURVE_PATH, HERE)}")
    # plt.show()   # 想在窗口里交互查看时取消注释


# ─── 7. 预测 ───────────────────────────────────────────────────────────
def predict(model_path, texts):
    """
    加载训练好的模型，对任意句子做语种识别（不需要再传 vocab / LABELS，都在 checkpoint 里）。

    打印 top-3 概率：只看第一名的概率不够信息量，
    看第二名能告诉我们"模型觉得这句话最像哪两种语言"，
    对 es/pt、zh/ja 这种近邻语种很有参考价值。

    :param model_path: main() 存下来的 .pt 文件
    :param texts: 待预测的句子列表
    """
    if not os.path.exists(model_path):
        # 第一次跑很容易忘：predict() 读的是训练产物，得先把 main() 的注释解开跑一次
        raise SystemExit(f"找不到模型文件：{model_path}\n"
                         f"  请先取消 __main__ 里 main() 的注释，训练一次再预测。")
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    vocab = ckpt["vocab"]
    labels = ckpt["labels"]

    model = LangIDClassifier(len(vocab), len(labels),
                             embed_dim=ckpt["embed_dim"],
                             hidden_dim=ckpt["hidden_dim"],
                             dropout=ckpt["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()    # 关掉 Dropout，否则同一个句子每次预测结果都可能不一样

    with torch.no_grad():
        for text in texts:
            ids = torch.tensor([encode(text, vocab, ckpt["maxlen"])])
            prob = torch.softmax(model(ids), dim=1)[0]
            top_prob, top_idx = prob.topk(3)
            parts = ", ".join(f"{labels[int(i)]}:{float(p):.3f}"
                              for p, i in zip(top_prob, top_idx))
            print(f"  {text[:40]}")
            print(f"      → {labels[int(top_idx[0])]}   （{parts}）")


if __name__ == "__main__":
    # main()   # 需要（重新）训练时取消注释；训练产物：langid_rnn_v12.pt + vocab_v12.txt + train_curve.png
    test_texts = [
        "我觉得一部电影比任何书都有趣。",                      # 中文
        "彼女は微笑みながら言いました。",                        # 日语
        "Мне нравится этот город.",                        # 俄语
        "저는 한국 음식을 아주 좋아합니다.",                    # 韩语（新增语种）
        "ฉันชอบอ่านหนังสือทุกวัน",                          # 泰语（新增语种）
        "Tôi thích uống cà phê vào buổi sáng.",            # 越南语（新增语种）
        "Bu akşam sinemaya gitmek istiyorum.",             # 土耳其语（新增语种）
        "This pull request 麻烦你明天 review 一下",           # 中英混排：模型会判成哪个？
    ]
    predict(MODEL_PATH, test_texts)
