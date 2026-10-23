# coding:utf-8
"""五分类任务：五维向量的最大值位于第几维，就属于第几类。

训练标签使用 0～4；展示时加 1，对应第 1～5 类。
最大值并列时，取最先出现的维度，与 torch.argmax 的规则一致。
"""

# 解决 OpenMP 库冲突问题
import os
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 5)  # 五个输出分别对应五个类别
        self.loss = nn.CrossEntropyLoss()

    # 传入标签时返回损失；否则返回五个类别的原始分数（logits）。
    def forward(self, x, y=None):
        logits = self.linear(x)  # (batch_size, 5)
        if y is not None:
            # CrossEntropyLoss 内部处理 softmax，不能提前对 logits 做归一化。
            return self.loss(logits, y)
        return logits


# 生成一个样本，例如 [0.1, 0.8, 0.2, 0.3, 0.4] 的训练标签为 1（第 2 类）。
def build_sample():
    x = torch.rand(5)
    return x, int(x.argmax().item())


# 向量的五个维度独立均匀采样，各类别的出现概率相同。
def build_dataset(total_sample_num):
    x = torch.rand(total_sample_num, 5)
    y = x.argmax(dim=1)  # shape 为 (样本数,)，类型为 torch.long
    return x, y


# 每轮使用独立于训练集的测试数据评估；传入数据可保持各轮评估集一致。
def evaluate(model, test_data=None):
    model.eval()
    x, y = build_dataset(1000) if test_data is None else test_data
    with torch.no_grad():
        predicted_classes = model(x).argmax(dim=1)
        correct = (predicted_classes == y).sum().item()
    accuracy = correct / y.numel()
    print("正确预测个数：%d/%d，正确率：%.2f%%" % (correct, y.numel(), accuracy * 100))
    return accuracy


def main(model_path="model.bin", show_plot=True):
    torch.manual_seed(42)  # 固定随机种子，便于复现实验
    epoch_num = 40
    batch_size = 100
    train_sample = 5000
    input_size = 5
    learning_rate = 0.03

    model = TorchModel(input_size)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_x, train_y = build_dataset(train_sample)
    test_data = build_dataset(1000)
    log = []

    for epoch in range(epoch_num):
        model.train()
        total_loss = 0.0
        # 每轮重新打乱样本顺序，并保留最后不足一个 batch 的样本。
        indices = torch.randperm(train_sample)
        for start in range(0, train_sample, batch_size):
            batch_indices = indices[start:start + batch_size]
            x = train_x[batch_indices]
            y = train_y[batch_indices]
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            total_loss += loss.item() * len(batch_indices)

        mean_loss = total_loss / train_sample
        print("=========\n第%d轮平均 loss：%f" % (epoch + 1, mean_loss))
        accuracy = evaluate(model, test_data)
        log.append([accuracy, mean_loss])

    torch.save(model.state_dict(), model_path)
    print("模型已保存到：%s" % model_path)

    if show_plot:
        import matplotlib.pyplot as plt

        epochs = range(1, epoch_num + 1)
        plt.plot(epochs, [item[0] for item in log], label="accuracy")
        plt.plot(epochs, [item[1] for item in log], label="loss")
        plt.xlabel("epoch")
        plt.legend()
        plt.show()
    return model


# 使用训练好的模型预测，支持单个五维向量或一批五维向量。
def predict(model_path, input_vec):
    x = torch.as_tensor(input_vec, dtype=torch.float32)
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if x.ndim != 2 or x.shape[1] != 5:
        raise ValueError("输入必须是五维向量，或由五维向量组成的列表")
    if not torch.isfinite(x).all():
        raise ValueError("输入向量不能包含 NaN 或无穷大")

    model = TorchModel(5)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(x), dim=1)
        confidence, predicted_indices = probabilities.max(dim=1)
        predicted_classes = predicted_indices + 1  # 转换为第 1～5 类

    for vec, category, probability in zip(x.tolist(), predicted_classes.tolist(), confidence.tolist()):
        print("输入：%s，预测类别：%d，概率值：%.4f" % (vec, category, probability))
    return predicted_classes.tolist()


if __name__ == "__main__":
    # 模型保存在脚本目录，直接运行即可先训练、再预测。
    model_path = Path(__file__).with_name("model.bin")
    main(model_path)
    test_vec = [
        [0.81889086, 0.15229675, 0.31082123, 0.03504317, 0.88920843],
        [0.94963533, 0.5524256, 0.95758807, 0.95520434, 0.84890681],
        [0.90797868, 0.67482528, 0.13625847, 0.34675372, 0.19871392],
        [0.39349776, 0.59416669, 0.92579291, 0.41567412, 0.8358894],
        [0.1, 0.9, 0.2, 0.3, 0.4],
        [0.1, 0.2, 0.3, 0.9, 0.4],
    ]
    predict(model_path, test_vec)
