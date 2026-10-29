# 第三周作业 · 12 语种识别

> 模仿课堂「8 语种识别」（`week3-深度学习常用组件/语种识别实战/train_langid_cls.py`），
> 将其扩展为 **12 语种识别**。数据来自第三周课件文件夹
> `week3-深度学习常用组件/语种识别实战/data/langid/v12`，已复制到本目录 `data/langid/v12/`。

## 一、任务

一句话 → 判断它属于 12 种语言中的哪一种（12 分类）。

- **12 语种** = 课内 8 语种 + 4 个新增：`ar en es hi ja ko pt ru th tr vi zh`
  （新增：`ko` 韩语、`th` 泰语、`vi` 越南语、`tr` 土耳其语）
- 模型：`Embedding → RNN → 掩码 MaxPool → LayerNorm → Dropout → Linear + Softmax`
- 损失：`CrossEntropyLoss`　优化器：`Adam (lr=1e-3)`
- 环境：`py312`（conda，torch 2.6.0+cpu），CPU 单线程即可

## 二、在课堂 8 语种代码上的三处改动

| # | 改动点 | 8 语种版 | 12 语种版 |
|---|--------|----------|----------|
| 1 | **LABELS** | 8 个（`ar en es hi ja pt ru zh`） | 12 个（新增 ko/th/vi/tr），模型输出维度随之变 12 |
| 2 | **数据目录** | `data/langid/v8` | `data/langid/v12` |
| 3 | **字符表** | 读随课件发的 `vocab_v8.txt` | **从 `v12/train.csv` 用 `build_char_vocab()` 自己生成**，写成 `vocab_v12.txt` |

> 课堂脚本注释里明确埋了伏笔：*"LABELS 直接列出了这 8 个语种…字符表不从训练集现算…要换数据集/加语种时这两处都得跟着改"* —— 本次作业正是把这三点落实。

## 三、文件清单

```
week03/
├── data/langid/
│   ├── v12/            # 12 语种数据（从课件复制）：train 9600 / dev 1200 / test 1800
│   │   ├── train.csv
│   │   ├── dev.csv
│   │   └── test.csv
│   ├── test_hard.csv   # 难度诊断集 1300 条（short/nodeaccent/nokana）
│   ├── test_openset.csv# 开放集 400 条（训练里没见过的 el/bg/sw/ur）
│   └── vocab/
│       └── vocab_v12.txt   # 作业自己生成的字符表（3276 字符）
├── train_langid_cls_v12.py  # 12 语种训练脚本（含自动生成字符表）
├── eval_langid_v12.py       # 12 语种评测脚本（test/难度集/开放集）
├── langid_rnn_v12.pt        # 训练好的模型权重
├── train_log.txt            # 训练日志
├── eval_test.txt            # 常规测试集评测日志
├── eval_hard.txt            # 难度集 + 开放集评测日志
└── README.md                # 本说明
```

## 四、运行方式

```bash
# 环境：conda 的 py312（torch 2.6.0+cpu）
"C:\Users\Corey1016\.conda\envs\py312\python.exe" -X utf8 train_langid_cls_v12.py --save langid_rnn_v12.pt

# 评测：常规测试集 / 难度集 / 开放集
... python -X utf8 eval_langid_v12.py --model langid_rnn_v12.pt
... python -X utf8 eval_langid_v12.py --model langid_rnn_v12.pt --split test_hard --openset
```

> 字符表逻辑：`train_langid_cls_v12.py` 启动时若 `vocab_v12.txt` 不存在，
> 会自动调用 `build_char_vocab()` 从 `v12/train.csv` 统计生成并写入文件（与 `vocab_v8.txt` 同格式）。

## 五、运行结果

### 1. 字符表
从 `v12/train.csv`（9600 条）生成：**3276 个字符** —— 与课件数据卡参考答案完全一致。

### 2. 训练
8 个 epoch，CPU 单线程 **8 秒**，模型参数量 **218,892**。最终 `dev_acc = 0.9642`。

### 3. 常规测试集（1800 条）
- **accuracy = 0.9533**　**macro-F1 = 0.9537**（随机猜仅 8.3%）
- 最易混淆：`pt→es`（36 条）、`en→es`（10）、`es→pt`（7）、`en→tr`（7）
- 逐语种：`ar/hi/ru/th` 满分；`pt` 最弱（0.740，与西葡同书写系统难分，和课堂观察一致）
- 按长度分档：8~12 字符 0.906 → 20~30 字符 0.975（短文本更难，符合预期）

### 4. 难度诊断集（1300 条）
| 切片 | 条数 | accuracy |
|---|---|---|
| short（2~7 字符短句） | 600 | 0.845 |
| nodeaccent（去重音） | 600 | 0.763 |
| nokana（日文去假名） | 100 | **0.260**（61 条被当成中文） |

> `nokana` 崩掉正是预期现象：去掉假名后日文只剩汉字，与中文几乎无法区分。

### 5. 开放集拒识（400 条未见语种 el/bg/sw/ur）
- 阈值（验证集 max-prob 5% 分位）= 0.698
- 正常输入保留率 95.0%，**开放集拒识率 38.7%**
- 被误认方向符合书写系统亲缘：`bg→ru`、`ur→ar`、`sw→tr/en`、`el→ja`

## 六、小结

12 语种识别在完全复用课堂模型结构与超参数的情况下即可达到 **~95%** 的测试准确率，
验证了「字符级 + 组件化模型」天然支持多语种扩展；困难主要来自同一书写系统内的
语言（西葡、土西）以及信息量极少的短文本，与课堂 8 语种版的结论一致。
