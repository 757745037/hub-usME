# -*- coding: utf-8 -*-
"""
lstm_forward_demo.py —— 手写 LSTM 前向，和 PyTorch 的 nn.LSTM 对齐

照 LSTMforward.py 的思路，但把重点放在"作业里为什么值得换成 LSTM"：

  1. 公式与 PyTorch 的权重排布（i, f, g, o 四块拼在一起）逐项对齐
  2. 三个门到底在干什么 —— 打印每一步的门控均值，看得见"记"和"忘"
  3. 反向传播的差别 —— 用同一条 40 步的链式结构，比较 RNN 与 LSTM
     在第 1 个时间步上拿到的梯度大小（梯度消失的直接证据）

LSTM 门控公式（PyTorch 约定顺序 i, f, g, o）:
  i_t = sigmoid(W_ii x_t + b_ii + W_hi h_{t-1} + b_hi)   输入门
  f_t = sigmoid(W_if x_t + b_if + W_hf h_{t-1} + b_hf)   遗忘门
  g_t =    tanh(W_ig x_t + b_ig + W_hg h_{t-1} + b_hg)   候选记忆
  o_t = sigmoid(W_io x_t + b_io + W_ho h_{t-1} + b_ho)   输出门
  c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t                        细胞状态
  h_t = o_t ⊙ tanh(c_t)                                  隐藏状态

运行：python lstm_forward_demo.py
"""

import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(42)
np.random.seed(42)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ─────────────────────────────────────────────────────────────────────────────
# 1. 手动实现单向 LSTM（和 LSTMforward.py 里的 DiyLSTM 同一套写法）
# ─────────────────────────────────────────────────────────────────────────────
class DiyLSTM:
    """
    weight_ih: [4*hidden, input]   按行分成 W_i, W_f, W_g, W_o 四块
    weight_hh: [4*hidden, hidden]  同上
    bias_ih / bias_hh: [4*hidden]  PyTorch 把偏置拆成 ih 和 hh 两组
    """

    def __init__(self, weight_ih, weight_hh, bias_ih, bias_hh, hidden_size):
        self.weight_ih = weight_ih
        self.weight_hh = weight_hh
        self.bias_ih = bias_ih
        self.bias_hh = bias_hh
        self.hidden_size = hidden_size

    def forward(self, x, trace=False):
        H = self.hidden_size
        h = np.zeros(H)
        c = np.zeros(H)
        output, gates_log = [], []

        for t, xt in enumerate(x):
            # 四个门一次性算完再切分，和 PyTorch 内部做法一致
            gates = (self.weight_ih @ xt + self.bias_ih
                     + self.weight_hh @ h + self.bias_hh)

            i = sigmoid(gates[0:H])            # 输入门：新信息写多少
            f = sigmoid(gates[H:2 * H])        # 遗忘门：老细胞状态留多少
            g = np.tanh(gates[2 * H:3 * H])    # 候选记忆
            o = sigmoid(gates[3 * H:])         # 输出门：细胞状态暴露多少

            c_prev = c
            c = f * c + i * g                  # 细胞状态：加法通路，梯度直通
            h = o * np.tanh(c)

            output.append(h.copy())
            gates_log.append((i.mean(), f.mean(), g.mean(), o.mean(),
                              float(np.abs(c).mean()),
                              float(np.abs(c_prev).mean())))

        if trace:
            return np.array(output), h, c, gates_log
        return np.array(output), h, c


# ─────────────────────────────────────────────────────────────────────────────
# 2. 逐项对齐 PyTorch
# ─────────────────────────────────────────────────────────────────────────────
def compare_with_torch():
    seq_len, input_size, hidden_size = 6, 5, 4
    x = np.random.randn(seq_len, input_size).astype(np.float32) * 0.5

    torch_lstm = nn.LSTM(input_size, hidden_size, bias=True, batch_first=True)
    sd = torch_lstm.state_dict()

    t_out, (t_hn, t_cn) = torch_lstm(torch.from_numpy(x)[None])
    t_out, t_hn, t_cn = t_out.detach().numpy()[0], t_hn.detach().numpy()[0], t_cn.detach().numpy()[0]

    diy = DiyLSTM(sd["weight_ih_l0"].numpy(), sd["weight_hh_l0"].numpy(),
                  sd["bias_ih_l0"].numpy(), sd["bias_hh_l0"].numpy(), hidden_size)
    d_out, d_h, d_c, gates = diy.forward(x, trace=True)

    print("=" * 66)
    print("【1】手写 LSTM 前向 vs PyTorch nn.LSTM")
    print("=" * 66)
    print(f"输入 shape：{x.shape}（seq_len=6, input=5），hidden={hidden_size}")
    print(f"PyTorch 第 0 步输出：{np.round(t_out[0], 4)}")
    print(f"DIY     第 0 步输出：{np.round(d_out[0], 4)}")
    print(f"PyTorch 末步输出  ：{np.round(t_out[-1], 4)}")
    print(f"DIY     末步输出  ：{np.round(d_out[-1], 4)}")
    print(f"\n输出最大误差：{np.abs(t_out - d_out).max():.2e}")
    print(f"h_n 最大误差：{np.abs(t_hn - d_h).max():.2e}")
    print(f"c_n 最大误差：{np.abs(t_cn - d_c).max():.2e}")
    print("→ 误差都在 1e-7 量级，说明四个门 + 细胞状态的顺序完全没错。\n")

    print("  每一步的门控均值（i 输入门 / f 遗忘门 / g 候选 / o 输出门）：")
    print(f"    {'t':>3}{'i':>9}{'f':>9}{'g':>9}{'o':>9}{'|c_t|':>9}{'|c_t-1|':>9}")
    for t, (i, f, g, o, c_now, c_prev) in enumerate(gates):
        print(f"    {t:>3}{i:>9.3f}{f:>9.3f}{g:>9.3f}{o:>9.3f}{c_now:>9.3f}{c_prev:>9.3f}")
    print("  · f 接近 1 → 细胞状态几乎原样往后传（长期记忆的通道）")
    print("  · i 接近 0 → 这一步的新信息基本不写入")
    print("  · c_t 的更新只做加法和数乘，不像 RNN 那样每一步都过 tanh 压缩\n")


# ─────────────────────────────────────────────────────────────────────────────
# 3. 梯度：同一条 40 步的链，RNN 与 LSTM 在第 1 步上各拿到多少梯度
#    做法：只让第 1 个时间步的输入带参数，损失取最后一步输出的平方和，
#          再看这个输入参数的梯度有多大 —— 梯度越小，说明"第一步被忘了"。
# ─────────────────────────────────────────────────────────────────────────────
def grad_through_time(cell="rnn", steps=40, hidden=16, forget_bias=0.0):
    """
    只有第 1 个时间步的输入可训练，中间全是 0，损失取最后一步输出的平方和。
    反过来求第 1 步输入的梯度 —— 它越大，说明"开头的信息"被记住并影响了输出。
    """
    torch.manual_seed(0)
    rnn = (nn.RNN if cell == "rnn" else nn.LSTM)(1, hidden, batch_first=True)
    if cell == "lstm" and forget_bias:
        # 把遗忘门偏置抬高：sigmoid(2)≈0.88，模拟"学出来的遗忘门"，
        # 这也是实践中常见的初始化技巧（PyTorch 默认是 0，此处只为把差别放大）
        with torch.no_grad():
            rnn.bias_ih_l0[hidden:2 * hidden].fill_(forget_bias)
            rnn.bias_hh_l0[hidden:2 * hidden].fill_(forget_bias)

    first = torch.zeros(1, 1, 1, requires_grad=True)
    x = torch.cat([first, torch.zeros(1, steps - 1, 1)], dim=1)

    out, _ = rnn(x)
    out[0, -1].pow(2).sum().backward()
    return first.grad.abs().mean().item()


def compare_gradient():
    print("=" * 66)
    print("【2】为什么 40 步的句子更该用 LSTM：梯度能传回第 1 步吗")
    print("=" * 66)
    print(f"    {'结构':<26}{'第1步输入的梯度':>18}")
    rows = [("RNN（课堂算法）", "rnn", 0.0),
            ("LSTM（遗忘门 bias=0）", "lstm", 0.0),
            ("LSTM（遗忘门 bias=2）", "lstm", 2.0)]
    for name, cell, fb in rows:
        g = grad_through_time(cell, forget_bias=fb)
        print(f"    {name:<26}{g:>18.3e}")
    print("  序列 40 步、中间 39 步输入全是 0，只有第 1 步带信息：")
    print("  · RNN 每步都要乘 W_hh 再过 tanh，40 次连乘 → 梯度归零，开头直接被忘掉；")
    print("  · LSTM 的细胞状态走 c_t = f⊙c_{t-1} + i⊙g 这条加法通路，")
    print("    f 停在 0.9 附近时衰减是 0.9^40≈0.015，而不是连乘一堆随机矩阵；")
    print("  · 所以同一句话里，LSTM 还记得 40 个字符之前发生了什么 —— 这正是语种识别")
    print("    需要的能力（一个字符像不像某语言，要结合前后文的搭配）。\n")


# ─────────────────────────────────────────────────────────────────────────────
# 4. 补齐位对双向 LSTM 的影响（作业里用了 pack_padded_sequence 的原因）
# ─────────────────────────────────────────────────────────────────────────────
def padding_effect():
    print("=" * 66)
    print("【3】[PAD] 混进双向 LSTM 会怎样")
    print("=" * 66)
    torch.manual_seed(0)
    lstm = nn.LSTM(4, 4, batch_first=True, bidirectional=True)

    real = torch.randn(1, 3, 4)                      # 真实句子 3 个字符
    padded = torch.cat([real, torch.zeros(1, 5, 4)], dim=1)   # 后面补 5 个 [PAD]

    full, _ = lstm(padded)                           # 不打包：倒着读先读到 PAD
    cut, _ = lstm(real)                              # 打包等价：只读真实长度

    print(f"  真实长度 3，补齐后长度 8")
    print(f"  不打包时第 3 步（真实末字符）反向输出前 4 维：{np.round(full[0, 2, 4:].detach().numpy(), 3)}")
    print(f"  打包后  第 3 步（真实末字符）反向输出前 4 维：{np.round(cut[0, 2, 4:].detach().numpy(), 3)}")
    print(f"  两者最大差异：{(full[0, :3, 4:] - cut[0, :3, 4:]).abs().max().item():.3f}")
    print("  不打包时反向分支从一串 0 起步，输出直接被污染；")
    print("  默认 40 步里中文句子常只有 10 来个字符，污染比例很高。\n")


if __name__ == "__main__":
    compare_with_torch()
    compare_gradient()
    padding_effect()
