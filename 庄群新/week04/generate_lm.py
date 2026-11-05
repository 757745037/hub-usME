# -*- coding: utf-8 -*-
"""
generate_lm.py —— 加载训练好的字符级 Transformer 语言模型，做续写 / 下一字联想。

参数说明：
    --topk        每步只在概率最高的前 K 个字里采样（1 = 贪心，永远选概率最大的字）
    --temperature 温度：越大越随机发散，越小越保守（趋近贪心）
    --max-new-tokens  最多续写多少个字
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────── 模型定义（与训练脚本保持完全一致）─────────────────
# 必须和 train_transformer_lm.py 中的类名、维度一致，否则 checkpoint 加载不上。


class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_ff, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, causal_mask):
        attn_out, _ = self.attn(x, x, x, attn_mask=causal_mask, need_weights=False)
        x = self.norm1(x + self.drop(attn_out))
        x = self.norm2(x + self.drop(self.ffn(x)))
        return x


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_ff, max_len, dropout):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_ff, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size)
        self.max_len = max_len

    def forward(self, x):
        B, T = x.shape
        positions = torch.arange(T, device=x.device)
        h = self.tok_emb(x) + self.pos_emb(positions)
        causal_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1
        )
        for block in self.blocks:
            h = block(h, causal_mask)
        h = self.norm(h)
        return self.fc(h)


# ─────────────────────────── 续写逻辑 ───────────────────────────────────────

def encode_prefix(text, char2idx, seq_len):
    """上文编码成 id 序列；超出 seq_len 时只保留最后 seq_len-1 个字。"""
    ids = [char2idx[c] for c in text if c in char2idx]
    if len(ids) > seq_len - 1:
        ids = ids[-(seq_len - 1):]
    return ids


@torch.no_grad()
def generate(model, char2idx, idx2char, prefix, max_new_tokens,
             topk, temperature, device, seq_len):
    """从 prefix 出发，逐字采样续写 max_new_tokens 个字，返回完整字符串。"""
    model.eval()
    ids = encode_prefix(prefix, char2idx, seq_len)
    if not ids:
        return prefix

    out = list(ids)
    for _ in range(max_new_tokens):
        x = torch.tensor([out[-seq_len:]], dtype=torch.long, device=device)
        logits = model(x)[0, -1, :] / temperature

        if topk > 1:
            topv, topi = torch.topk(logits, min(topk, logits.numel()))
            probs = F.softmax(topv, dim=-1)
            next_id = int(topi[torch.multinomial(probs, 1)])
        else:
            next_id = int(logits.argmax())

        out.append(next_id)

    return prefix + "".join(idx2char[i] for i in out[len(ids):])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="transformer_lm.pt")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=30)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.model, map_location=device)
    char2idx = ckpt["char2idx"]
    idx2char = ckpt["idx2char"]
    cfg = ckpt["args"]

    model = TransformerLM(
        vocab_size=len(char2idx),
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        dim_ff=cfg["dim_ff"],
        max_len=cfg["seq_len"],
        dropout=0.0,   # 推理阶段关闭 dropout，保证输出确定性
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print("=" * 52)
    print("  字符级 Transformer 语言模型 · 续写")
    print("  输入上文回车 → 模型逐字续写；q 退出")
    print("=" * 52)
    print(f"  模型: {args.model}  词表 {len(char2idx)} 字  "
          f"topk={args.topk}  temp={args.temperature}")

    while True:
        prefix = input("\n上文> ").strip()
        if prefix.lower() == "q":
            print("退出。")
            break
        if not prefix:
            continue

        text = generate(model, char2idx, idx2char, prefix,
                        args.max_new_tokens, args.topk, args.temperature,
                        device, cfg["seq_len"])
        print(f"续写> {text}")


if __name__ == "__main__":
    main()
