# coding: utf-8
"""仿照 TorchDemo.py：使用 Softmax 完成五维向量的五分类训练。

规律：最大数在第几维，就属于第几类。
例如 [0.1, 0.3, 0.8, 0.2, 0.5] 属于第 3 类。
内部标签为 0～4，输出给用户的类别为 1～5。
最大值并列时，生成标签采用第一次出现的位置。

训练得到的是近似分类器，接近或并列的最大值仍可能被误分类。
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


OUTPUT_DIR = Path(__file__).resolve().parent
MODEL_PATH = OUTPUT_DIR / "softmax_model.bin"


class TorchModel(nn.Module):
    def __init__(self, input_size=5):
        super().__init__()
        self.linear = nn.Linear(input_size, 5)  # 五个类别，对应五个输出分数
        self.activation = nn.Softmax(dim=-1)  # 对每个样本的五个类别归一化
        self.loss = nn.CrossEntropyLoss()

    # 输入真实标签时返回 loss，否则返回五个类别的概率
    def forward(self, x, y=None):
        # 每个向量独立归一化，保持各维的大小顺序。
        # 先缩放再减最小值，以适应负数和不同数值范围。
        scale = x.abs().amax(dim=-1, keepdim=True)
        x = x / torch.where(scale == 0, torch.ones_like(scale), scale)
        x = x - x.amin(dim=-1, keepdim=True)
        span = x.amax(dim=-1, keepdim=True)
        x = x / torch.where(span == 0, torch.ones_like(span), span)

        logits = self.linear(x.to(self.linear.weight.dtype))  # (batch_size, 5)
        if y is not None:
            # CrossEntropyLoss 内部包含 LogSoftmax；输入原始分数即可。
            # 不要把已经 Softmax 的概率再传给 CrossEntropyLoss。
            return self.loss(logits, y)
        return self.activation(logits)


# 生成一个样本：argmax 只用来生成真实标签，不替代模型预测。
def build_sample():
    x = np.random.random(5).astype(np.float32)
    y = int(np.argmax(x))  # 最大值的下标，取值为 0、1、2、3、4
    return x, y


def build_dataset(total_sample_num):
    X, Y = [], []
    for _ in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)  # 多分类标签是一维类别编号，不需要 [y] 或 one-hot
    return torch.from_numpy(np.asarray(X)), torch.tensor(Y, dtype=torch.long)


def evaluate(model, x, y):
    model.eval()
    with torch.no_grad():
        probabilities = model(x)
        predicted_labels = probabilities.argmax(dim=1)
        correct = (predicted_labels == y).sum().item()
    accuracy = correct / len(y)
    print("正确预测：%d / %d，准确率：%.2f%%" % (correct, len(y), accuracy * 100))
    return accuracy


def main():
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(1)  # 小模型使用单线程，减少线程调度开销

    epoch_num = 30
    batch_size = 64
    train_sample = 10000
    learning_rate = 0.01

    model = TorchModel(input_size=5)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_x, train_y = build_dataset(train_sample)
    valid_x, valid_y = build_dataset(2000)  # 固定验证集，用于观察每轮效果
    log = []

    for epoch in range(epoch_num):
        model.train()
        indices = torch.randperm(train_sample)  # 每轮打乱训练样本顺序
        total_loss = 0.0
        for start in range(0, train_sample, batch_size):
            batch_indices = indices[start:start + batch_size]
            x, y = train_x[batch_indices], train_y[batch_indices]
            optim.zero_grad()  # 梯度清零
            loss = model(x, y)  # 前向传播，计算交叉熵
            loss.backward()  # 反向传播，计算梯度
            optim.step()  # 更新模型参数
            total_loss += loss.item() * len(batch_indices)

        average_loss = total_loss / train_sample
        print("第 %02d/%d 轮，平均 loss：%.6f" % (epoch + 1, epoch_num, average_loss))
        accuracy = evaluate(model, valid_x, valid_y)
        log.append({"epoch": epoch + 1, "loss": average_loss, "accuracy": accuracy})

    # 最后使用独立生成的测试集评估，不参与梯度更新。
    test_x, test_y = build_dataset(10000)
    print("\n独立测试集：")
    test_accuracy = evaluate(model, test_x, test_y)
    torch.save(model.state_dict(), MODEL_PATH)
    (OUTPUT_DIR / "softmax_training_log.json").write_text(
        json.dumps({"epochs": log, "test_accuracy": test_accuracy}, indent=2),
        encoding="utf-8",
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = [item["epoch"] for item in log]
    axes[0].plot(epochs, [item["loss"] for item in log])
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Training loss")
    axes[1].plot(epochs, [item["accuracy"] for item in log])
    axes[1].set(xlabel="Epoch", ylabel="Accuracy", title="Validation accuracy", ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "softmax_training_curve.png", dpi=150)
    print("模型已保存：%s" % MODEL_PATH)
    print("训练曲线已打开，关闭图像窗口后继续运行预测示例。")
    plt.show()  # 与原示例一致：弹出 loss 和准确率曲线窗口
    plt.close(fig)
    return model


# 支持输入一个五维向量，也支持一次输入多个五维向量。
def predict(model_path, input_vec):
    x = torch.as_tensor(input_vec, dtype=torch.float64)
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] != 5:
        raise ValueError("请输入一个五维向量，或由多个五维向量组成的列表")
    if not torch.isfinite(x).all():
        raise ValueError("向量中的数字必须是有限值，不能包含 NaN 或无穷大")

    model = TorchModel(input_size=5)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    with torch.no_grad():
        probabilities = model(x)
        classes = probabilities.argmax(dim=1) + 1  # 将下标 0～4 转成第 1～5 类

    for vec, probability, category in zip(x.tolist(), probabilities, classes.tolist()):
        print("输入：%s" % vec)
        print("预测：第 %d 类；五类概率：%s" % (
            category, [round(value, 4) for value in probability.tolist()]
        ))
    return classes, probabilities


if __name__ == "__main__":
    main()
    print("\n加载训练好的模型，预测示例：")
    test_vec = [
        [0.9, 0.2, 0.3, 0.4, 0.5],  # 第 1 类
        [0.1, 0.9, 0.3, 0.4, 0.5],  # 第 2 类
        [0.1, 0.2, 0.9, 0.4, 0.5],  # 第 3 类
        [0.1, 0.2, 0.3, 0.9, 0.5],  # 第 4 类
        [0.1, 0.2, 0.3, 0.4, 0.9],  # 第 5 类
        [-8, -2, -5, -9, -4],       # 第 2 类，全部为负数
        [10, 20, 30, 90, 50],       # 第 4 类，较大的数值范围
    ]
    predict(MODEL_PATH, test_vec)

