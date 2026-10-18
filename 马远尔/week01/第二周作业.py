import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1. 超参数
# ============================================================
SEED = 42
TEXT_LENGTH = 5
NUM_CLASSES = 5
N_SAMPLES = 5000
TRAIN_RATIO = 0.8

EMBED_DIM = 32
HIDDEN_DIM = 64
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10

random.seed(SEED)
torch.manual_seed(SEED)


# 注意：这个字符池中不能包含“你”，否则一条文本可能出现两个“你”。
OTHER_CHARS = list(
    "的一是了不在人有我他这中大来上个们到说国和地也子时道出而要于就下得可"
    "生会自着去之过家学对能里后小么心多天发成法现所作用方主样理新开行分经动面起看定"
)


# ============================================================
# 2. 生成数据
# ============================================================
def make_sample(label):
    """生成一条五字文本；label 0～4 对应“你”在第1～5位。"""
    chars = [random.choice(OTHER_CHARS) for _ in range(TEXT_LENGTH)]
    chars[label] = "你"
    text = "".join(chars)
    return text, label


def build_dataset(n_samples=N_SAMPLES):
    """生成类别数量均衡的五分类数据集。"""
    data = []

    for i in range(n_samples):
        label = i % NUM_CLASSES
        text, label = make_sample(label)
        data.append((text, label))

    random.shuffle(data)
    return data


# ============================================================
# 3. 建立词表、编码文本
# ============================================================
def build_vocab(data):
    vocab = {
        "<PAD>": 0,
        "<UNK>": 1,
    }

    for text, _ in data:
        for char in text:
            if char not in vocab:
                vocab[char] = len(vocab)

    return vocab


def encode_text(text, vocab):
    ids = []

    for char in text:
        char_id = vocab.get(char, vocab["<UNK>"])
        ids.append(char_id)

    return ids


# ============================================================
# 4. PyTorch Dataset
# ============================================================
class YouPositionDataset(Dataset):
    def __init__(self, data, vocab):
        self.data = data
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text, label = self.data[index]
        ids = encode_text(text, self.vocab)

        x = torch.tensor(ids, dtype=torch.long)
        y = torch.tensor(label, dtype=torch.long)
        return x, y


# ============================================================
# 5. RNN模型
# ============================================================
class YouPositionRNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=NUM_CLASSES,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0,
        )

        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        self.fc = nn.Linear(
            in_features=hidden_dim,
            out_features=num_classes,
        )

    def forward(self, x):
        embedded = self.embedding(x)       # (B, 5, 32)
        _, hidden = self.rnn(embedded)     # hidden: (1, B, 64)
        last_hidden = hidden[-1]           # (B, 64)
        logits = self.fc(last_hidden)      # (B, 5)
        return logits


# ============================================================
# 6. 验证函数
# ============================================================
def evaluate(model, data_loader, criterion):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            predictions = logits.argmax(dim=1)

            batch_size = batch_x.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (predictions == batch_y).sum().item()
            total_samples += batch_size

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return average_loss, accuracy


# ============================================================
# 7. 主程序：准备数据、创建模型并训练
# ============================================================
def main():
    print("PyTorch版本：", torch.__version__)
    print("分类数量：", NUM_CLASSES)

    # 7.1 只生成一次完整数据集
    data = build_dataset()

    # 7.2 划分训练集和验证集
    split_index = int(len(data) * TRAIN_RATIO)
    train_data = data[:split_index]
    val_data = data[split_index:]

    # 7.3 只使用训练集建立词表
    vocab = build_vocab(train_data)

    print("完整数据集大小：", len(data))
    print("训练集大小：", len(train_data))
    print("验证集大小：", len(val_data))
    print("词表大小：", len(vocab))

    # 检查前5条数据，确保每条数据只有一个“你”，且标签正确。
    print("\n前5条数据：")
    for text, label in data[:5]:
        assert len(text) == TEXT_LENGTH
        assert text.count("你") == 1
        assert text[label] == "你"
        print(text, "标签：", label, "实际位置：第", label + 1, "位")

    # 7.4 创建Dataset和DataLoader
    train_dataset = YouPositionDataset(train_data, vocab)
    val_dataset = YouPositionDataset(val_data, vocab)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # 7.5 创建模型
    model = YouPositionRNN(vocab_size=len(vocab))

    # 7.6 必须在训练循环之前创建损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    print("\n模型结构：")
    print(model)
    print("\n开始训练RNN模型……")

    # 7.7 训练循环
    for epoch in range(1, EPOCHS + 1):
        model.train()

        total_train_loss = 0.0
        total_train_correct = 0
        total_train_samples = 0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()

            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            loss.backward()
            optimizer.step()

            predictions = logits.argmax(dim=1)
            batch_size = batch_x.size(0)

            total_train_loss += loss.item() * batch_size
            total_train_correct += (predictions == batch_y).sum().item()
            total_train_samples += batch_size

        train_loss = total_train_loss / total_train_samples
        train_accuracy = total_train_correct / total_train_samples

        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            criterion,
        )

        print(
            f"Epoch {epoch:2d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_accuracy:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_accuracy:.4f}"
        )


if __name__ == "__main__":
    main()
