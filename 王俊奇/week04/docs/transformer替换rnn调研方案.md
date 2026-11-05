# 调研方案：使用 Transformer 替换 RNN 字符级语言模型

## 一、背景与现状

现有工程由两个脚本组成，共用同一套 `LM` 模型定义：

1. `language_model.py`：字符级语言模型训练脚本，支持 RNN / LSTM 切换，含 PPL 计算。
2. `pinyin_ime.py`：基于训练好的 checkpoint 做拼音输入法，内部复制了一份相同的 `LM` 类用于加载权重。

现有模型结构（两个文件中保持一致，才能正确加载 checkpoint）：

```
Embedding(vocab, embed_dim)
→ Dropout
→ nn.RNN / nn.LSTM(embed_dim, hidden_dim, num_layers)
→ Dropout
→ Linear(hidden_dim, vocab)
```

推理方式：束搜索时把 `prefix + partial` 整段送入模型，取最后一个时间步 logits 做下一字打分。

## 二、改造目标

- 新增 Transformer 版语言模型，与现有 RNN/LSTM 通过 `--model` 参数共存（不做旧兼容处理，训练时明确指定即可）。
- 保持数据集 `CharDataset`、训练/评估循环 `run_epoch`、束搜索 `beam_search` 的接口不变，只替换模型与必要的超参。
- 训练保存的 checkpoint 结构不变（`model_state / char2idx / idx2char / args`），`pinyin_ime.py` 能依据 `args` 自动构建对应模型加载。

## 三、技术方案

### 3.1 模型结构（新增 `TransformerLM`）

```
Embedding(vocab, d_model)
+ 位置编码（正弦位置编码，无需训练参数，序列长短均可外推到训练长度内）
→ Dropout
→ N × nn.TransformerEncoderLayer（自注意力 + 前馈 + 残差 + LayerNorm）
→ Dropout
→ Linear(d_model, vocab)
```

关键点：

- 因果掩码（causal mask）：语言模型只能看到上文。使用 `nn.TransformerEncoder` 的 `mask` 参数传入上三角掩码（`bool` 型，`True` 表示禁止注意的位置），避免用 `float` 掩码触发 PyTorch 新版的嵌套张量性能警告。
- PyTorch 的 `TransformerEncoderLayer` 默认 `batch_first=False`，必须显式传 `batch_first=True`，否则维度含义颠倒。
- `d_model` 与 embedding 维度必须一致；`dim_feedforward` 取 `4 * d_model`（标准配置）。
- 单层时同样不允许设置 `dropout`，显式置 0（与现有 RNN 代码同思路）。

### 3.2 参数与命令行

新增/调整参数（`language_model.py`）：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | 取值扩展为 `rnn / lstm / transformer` | `lstm` |
| `--nhead` | Transformer 注意力头数，需能整除 `embed_dim` | `4` |
| `--dim_feedforward` | 前馈层维度 | `512`（约 4 倍 embed_dim） |
| `--max_len` | 位置编码支持的最大序列长度 | `512` |

`embed_dim`、`num_layers`、`dropout` 复用现有参数。`hidden_dim` 仅对 RNN/LSTM 有意义，Transformer 下忽略（不删除，避免改动 RNN 路径）。

### 3.3 checkpoint 与 pinyin_ime.py 的适配

- 训练保存 `args` 后，`pinyin_ime.py` 根据 `cfg["model"]` 分支构建 `TransformerLM` 或现有 `LM`：
  - Transformer 分支读取 `cfg["embed_dim"] / cfg["num_layers"] / cfg["nhead"] / cfg["dim_feedforward"]`，dropout 置 0（推理确定性，与现有做法一致）。
- 推理打分逻辑不变：整段 `prefix + partial` 输入，取最后一位 logits 做 `log_softmax`。因果掩码保证了 Transformer 在这一用法下与 RNN 行为一致（每步输出只依赖其左侧上文）。
- `beam_search`、`segment`、字频先验、交互循环全部不动。

### 3.4 上下文长度处理

训练时 `seq_len` 即注意力窗口；推理时若 `prefix + partial` 超过训练长度，只取最后 `seq_len` 个字符作为上文（与 RNN 的有限记忆语义一致，束搜索中拼接串本来就短，通常不会触发）。

## 四、超参建议（课堂语料规模）

- `embed_dim = 128`、`nhead = 4`、`num_layers = 2`、`dim_feedforward = 512`、`dropout = 0.1`。
- 学习率建议降为 `3e-4`：Transformer 对学习率更敏感，`1e-3` 配 Adam 容易训练不稳定（PPL 不降或震荡）。
- `seq_len = 64` 可保持不变；如需对比可再试 `seq_len = 128`（Transformer 处理长依赖的优势场景）。

## 五、改动清单（实施时执行）

1. `language_model.py`
   - 新增 `PositionalEncoding` 类与 `TransformerLM` 类。
   - `main()` 中按 `--model` 分支构建模型；新增 `--nhead / --dim_feedforward / --max_len` 参数。
   - `TransformerLM` 的 forward 需生成因果掩码并传给 encoder。
2. `pinyin_ime.py`
   - `main()` 中模型构建处增加 Transformer 分支（从 `cfg` 读取新超参）。
3. 训练验证：
   - `python language_model.py --model transformer --epochs 20`
   - 对比 `best_model.pt`（RNN/LSTM）与 `best_model_transformer.pt` 的验证 PPL。
4. 推理验证：
   - `python pinyin_ime.py --model_path best_model_transformer.pt`，确认候选输出正常、多轮上文连续性生效。

## 六、风险与取舍

- Transformer 无 RNN 的"无限"状态记忆，超出窗口的上文直接截断；本场景上下文短，影响可忽略。
- 小语料 + 小模型下，Transformer 收敛可能不如 LSTM 稳定，需用较低学习率；这是调参问题而非结构问题。
- 参数量对比：同维度下 Transformer 前馈层（4 倍 d_model）会带来更多参数，若过拟合可下调 `dim_feedforward`。
