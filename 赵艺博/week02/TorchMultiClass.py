# coding:utf8

# 解决 OpenMP 库冲突问题
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""
基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务

规律：x是一个5维向量，它最大数字在第几维就属于第几类
例如：[0.1, 0.9, 0.3, 0.2, 0.5] -> 最大值0.9在第1维 -> 类别1
共5类，类别编号为 0, 1, 2, 3, 4
"""


class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)  # 线性层：5维输入 -> 5类输出
        # CrossEntropyLoss 内部已包含 Softmax，训练时无需再手动做激活
        self.loss = nn.CrossEntropyLoss()

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        y_pred = self.linear(x)  # (batch_size, input_size) -> (batch_size, num_classes)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算交叉熵损失
        else:
            return y_pred  # 输出各类别的logits


# 生成一个样本
# 随机生成一个5维向量，最大值所在的维度即为类别
def build_sample():
    x = np.random.random(5)
    y = np.argmax(x)  # 最大值所在的下标，即类别 0~4
    return x, y


# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    # 标签使用 LongTensor，供 CrossEntropyLoss 使用
    return torch.FloatTensor(np.array(X)), torch.LongTensor(np.array(Y))


# 测试代码：用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 200
    x, y = build_dataset(test_sample_num)
    # 统计各类样本数量
    class_count = [0] * 5
    for label in y:
        class_count[int(label)] += 1
    print("本次预测集中各类样本数：", class_count)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # (batch_size, 5) logits
        y_pred_class = torch.argmax(y_pred, dim=1)  # 取概率最大的类别
        for y_p, y_t in zip(y_pred_class, y):
            if int(y_p) == int(y_t):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 20       # 训练轮数
    batch_size = 20      # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5       # 输入向量维度
    num_classes = 5      # 分类类别数
    learning_rate = 0.01 # 学习率
    # 建立模型
    model = TorchModel(input_size, num_classes)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model(x, y)   # 计算loss
            loss.backward()      # 计算梯度
            optim.step()         # 更新权重
            optim.zero_grad()    # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    model_path = os.path.join(os.path.dirname(__file__), "model.bin")
    torch.save(model.state_dict(), model_path)
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.title("Multi-class: argmax of 5D vector")
    fig_path = os.path.join(os.path.dirname(__file__), "train_curve.png")
    plt.savefig(fig_path)
    try:
        plt.show()
    except Exception:
        pass
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    num_classes = 5
    model = TorchModel(input_size, num_classes)
    model.load_state_dict(torch.load(model_path))
    print(model.state_dict())

    model.eval()
    with torch.no_grad():
        result = model.forward(torch.FloatTensor(input_vec))  # logits
        # Softmax 得到各类概率，argmax 得到预测类别
        prob = torch.softmax(result, dim=1)
        pred_class = torch.argmax(prob, dim=1)
    for vec, cls, p in zip(input_vec, pred_class, prob):
        true_cls = int(np.argmax(vec))
        print("输入：%s, 真实类别：%d, 预测类别：%d, 各类概率：%s"
              % (vec, true_cls, int(cls), p.numpy().round(4)))


if __name__ == "__main__":
    main()
    # 训练完成后做几组预测验证
    test_vec = [
        [0.81889086, 0.15229675, 0.31082123, 0.03504317, 0.88920843],  # max在第4维 -> 类4
        [0.94963533, 0.5524256, 0.95758807, 0.95520434, 0.84890681],   # max在第2维 -> 类2
        [0.90797868, 0.67482528, 0.13625847, 0.34675372, 0.19871392],  # max在第0维 -> 类0
        [0.39349776, 0.59416669, 0.92579291, 0.41567412, 0.8358894],   # max在第2维 -> 类2
        [0.1, 0.9, 0.3, 0.2, 0.5],                                      # max在第1维 -> 类1
    ]
    model_path = os.path.join(os.path.dirname(__file__), "model.bin")
    predict(model_path, test_vec)
