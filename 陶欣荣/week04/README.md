# Week04 字符级 Transformer 语言模型

本目录保存已训练 10 轮的同规模 Transformer，参数量为 1,770,378，最佳权重来自第 10 轮。

| 文件 | 用途 |
| --- | --- |
| best_model_transformer_matched.pt | 新 Transformer 权重，内含字表、超参数、最佳轮次和语料指纹 |
| best_model_transformer_matched.metrics.json | 逐轮训练与验证 loss、PPL 和训练配置 |
| test_model.py | 一键加载模型并复评验证集 PPL，同时评估旧 RNN |
| language_model.py | 模型定义、语料处理、训练与 PPL 评估 |
| pinyin_ime.py | 交互式拼音输入测试，默认加载本目录的新 Transformer |
| corpus_general.txt | 训练及验证使用的字符语料 |
| pinyin_map.json | 拼音到候选汉字的映射 |
| vocab_stats.json | 4000 字模型字表及完整语料的字频、频率、覆盖率 |
| best_model_rnn.pt | 旧 RNN 对照权重 |

## 运行环境

已验证 Python 3.12 和 PyTorch 2.6.0 CPU 版。本机可使用 Conda 的 py312 环境，解释器位于 D:/anaconda3/envs/py312/python.exe。

在 PowerShell 中运行以下命令。

~~~powershell
conda activate py312
cd "D:/pythontext/循环神经网络语言模型/week04"
python test_model.py
~~~

PPL 按所有有效目标字符的平均交叉熵取指数计算。默认复评 301,184 个验证目标字符，预期 Transformer PPL 约为 34.61，旧 RNN 约为 40.90，PPL 越低越好。

输入法测试可运行以下命令，然后输入 zhong guo 等拼音。

~~~powershell
python pinyin_ime.py
~~~

## 重现训练

以下参数对应已保存的新模型，训练产生新的权重及逐轮指标文件。

~~~powershell
python language_model.py --baseline best_model_rnn.pt --save best_model_transformer_retrained.pt --epochs 10 --batch_size 128 --embed_dim 192 --hidden_dim 516 --heads 4 --num_layers 2 --seq_len 64 --stride 64 --dropout 0.3 --lr 0.001
~~~

## 评估口径

沿用旧 RNN 的 4000 字词表和编号，随机种子为 42，按行打乱后取 95% 作为训练集、5% 作为验证集，序列长度为 64。词表外字符按原项目逻辑过滤。

当前语料重建出的字表与旧权重存在差异，训练时通过 --baseline 直接读取旧字表。旧权重未保存原训练语料的指纹，因此其历史语料版本无法严格核实。这里报告的是两个权重在当前同一验证集上的分数。

验证集中约 60.1% 的行也出现在训练集中。以上 PPL 用于当前验证集对照，对全新句子的泛化效果需要独立测试。旧 RNN 配置记录了计划训练 10 轮，其最佳轮次未保存。
