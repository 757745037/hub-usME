# 解决OpenMP多线程冲突，规避Anaconda环境下的线程报错
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""
任务描述：多分类任务
输入：长度为5的一维随机向量
分类规则：向量哪一个下标数值最大，样本类别就等于该下标（0-4，一共5类）
"""


class TorchModel(nn.Module):
    """神经网络模型类"""

    def __init__(self, input_size):
        # 调用父类构造函数
        super().__init__()
        # 单层全连接层：输入5维特征，输出5个类别的原始得分
        self.linear = nn.Linear(input_size, 5)
        self.activation = torch.sigmoid
        # 交叉熵损失函数，内部自带softmax处理，训练时不再需要了
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        """
        前向传播
        :param x: 输入特征张量 [batch_size,5]
        :param y: 真实标签；y=None代表预测模式，有值代表训练模式
        :return: 训练返回损失；预测返回各类别概率
        """
        logits = self.linear(x)  # 获取未经过激活的原始分数
        y_pred = self.activation(logits)

        if y is not None:
            # 训练阶段，直接将原始得分送入交叉熵损失
            return self.loss(y_pred, y)
        else:
            # 推理阶段，沿类别维度dim=1计算softmax概率
            return y_pred


def build_sample():
    """生成单条样本"""
    x = np.random.random(5)  # 生成0~1之间、长度为5的随机数组
    max_index = np.argmax(x)  # 最大值对应的下标即为标签
    return x, max_index


def build_dataset(total_sample_num):
    """批量构建数据集"""
    X = []  # 存储所有特征向量
    Y = []  # 存储所有样本标签
    for _ in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    # 先转为numpy数组再转为张量，消除列表ndarray转tensor的性能警告（有警告报错，但是不影响结果）
    X = np.array(X)
    Y = np.array(Y)
    # 特征是小数数值，使用浮点张量FloatTensor
    # 多分类cross_entropy损失要求类别标签必须为整数下标，因此使用LongTensor（int64），不可用float
    return torch.FloatTensor(X), torch.LongTensor(Y)


def evaluate(model):
    """模型评估函数，返回测试集准确率"""
    model.eval()  # 切换至评估模式
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    # 统计5个类别样本数量
    counts = np.bincount(y.numpy(), minlength=5)
    print(f"测试集各类样本数量：0:{counts[0]},1:{counts[1]},2:{counts[2]},3:{counts[3]},4:{counts[4]}")

    correct = 0
    with torch.no_grad():  # 评估关闭梯度计算，节省资源
        prob = model(x)
        pred = torch.argmax(prob, dim=1)  # 取概率最大下标当作预测类别
        correct = torch.sum(pred == y).item()

    acc = correct / test_sample_num
    print(f"预测正确数量:{correct}, 正确率:{acc:.4f}")


def main():
    """模型训练入口"""
    epoch_num = 60  # 整体数据集迭代轮数（由20增加到60，调参解决测试预测时有出现不对的情况）
    batch_size = 40  # 单批次样本数量
    train_sample = 15000  # 训练集样本总数（由5000增加到15000，调参解决测试预测时有出现不对的情况）
    input_size = 5  # 输入特征维度
    lr = 0.01  # 学习率

    model = TorchModel(input_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    log = []  # 保存每轮准确率、平均损失，用于绘图

    # 生成全部训练数据
    train_x, train_y = build_dataset(train_sample)

    for epoch in range(epoch_num):
        model.train()  # 切换为训练模式
        loss_list = []
        # 遍历每一个批次
        for batch_idx in range(train_sample // batch_size):
            start = batch_idx * batch_size
            end = start + batch_size
            x_batch = train_x[start:end]
            y_batch = train_y[start:end]

            loss = model(x_batch, y_batch)  # 前向传播求损失
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新权重参数
            optimizer.zero_grad()  # 清空梯度，防止梯度累积
            loss_list.append(loss.item())

        avg_loss = np.mean(loss_list)
        print(f"====第{epoch + 1}轮,平均loss:{avg_loss:.5f}====")
        acc = evaluate(model)
        log.append([acc, avg_loss])

    torch.save(model.state_dict(), "model.bin")  # 保存训练权重
    # 绘制准确率、损失曲线
    plt.plot([item[0] for item in log], label="accuracy")
    plt.plot([item[1] for item in log], label="loss")
    plt.legend()
    plt.show()


def predict(model_path, input_vec):
    """加载权重，对自定义向量进行预测"""
    model = TorchModel(5)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        prob = model(torch.FloatTensor(input_vec))
        res = torch.argmax(prob, dim=1)

    for vec, cls, p in zip(input_vec, res, prob):
        print(f"输入:{vec} → 预测类别:{int(cls)}, 各类概率{np.round(p.numpy(), 3)}")


if __name__ == "__main__":
    # main()  # 需要重新训练时取消注释
    test_vec = [
        [0.88889086, 0.15229675, 0.31082123, 0.03504317, 0.78920843],
        [0.94963533, 0.5524256, 0.95758807, 0.95520434, 0.84890681],
        [0.90797868, 0.67482528, 0.13625847, 0.34675372, 0.19871392],
        [0.99349776, 0.59416669, 0.92579291, 0.41567412, 0.1358894]
    ]
    predict("model.bin", test_vec)
