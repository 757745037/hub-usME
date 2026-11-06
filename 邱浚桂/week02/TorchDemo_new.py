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
实现一个自行构造的找规律(机器学习)任务 —— 多分类版本

规律（三分类）：x 是一个5维向量
    1) 如果最大值出现在第5个维度(下标4)，则为【正样本】 -> 类别 2
    2) 反之，如果最小值出现在第1个维度(下标0)，则为【负样本】 -> 类别 0
    3) 其余情况为【中性样本】                            -> 类别 1

与 TorchDemo.py（二分类 + mse_loss）的区别：
    1) 输出层由 1 个神经元改为 class_num(3) 个神经元，输出每个类别的"得分(logits)"
    2) 去掉 sigmoid，因为交叉熵损失函数内部自带 softmax(LogSoftmax)
    3) 损失函数由 mse_loss 改为 cross_entropy（参考 CrossEntropy.py）
    4) 标签 y 由 FloatTensor(n,1) 改为 LongTensor(n)，存的是类别编号而不是0/1概率
    5) 预测时由"是否大于0.5"改为"取概率最大的类别 argmax"

"""


class TorchModel(nn.Module):
    def __init__(self, input_size, class_num, hidden_size=0):
        super(TorchModel, self).__init__()
        self.class_num = class_num
        # hidden_size=0 时就是纯线性模型(与TorchDemo.py一致)；>0 时多一层非线性，拟合能力更强
        if hidden_size > 0:
            self.layer1 = nn.Linear(input_size, hidden_size)
            self.layer2 = nn.Linear(hidden_size, class_num)
            self.activation = torch.relu
        else:
            self.layer1 = None
            self.linear = nn.Linear(input_size, class_num)  # 线性层 5 -> class_num
        # 注意：多分类交叉熵要求输入是未经过softmax的logits，
        # nn.functional.cross_entropy = log_softmax + nll_loss，内部会自动做归一化
        self.loss = nn.functional.cross_entropy

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        if self.layer1 is not None:
            x = self.activation(self.layer1(x))
            y_pred = self.layer2(x)   # (batch_size, hidden_size) -> (batch_size, class_num)
        else:
            y_pred = self.linear(x)   # (batch_size, input_size) -> (batch_size, class_num)
        if y is not None:
            # cross_entropy 要求 y 是一维的 LongTensor，取值为 [0, class_num-1]
            return self.loss(y_pred, y)
        else:
            return y_pred  # 输出每个类别的得分(logits)


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量：
#   最大值在第5维 -> 正样本(类别2)
#   最小值在第1维 -> 负样本(类别0)
#   其余          -> 中性样本(类别1)
# 注意：两个条件可能同时满足，这里按"正样本优先"的规则判定
def build_sample():
    x = np.random.random(5)
    if np.argmax(x) == 4:          # 最大值是第5维
        return x, 2
    elif np.argmin(x) == 0:        # 最小值是第1维
        return x, 0
    else:
        return x, 1


# 随机生成一批样本
# 由于"中性样本"天然占比最高(约65%)，这里做类别均衡采样，保证三类数量一致，
# 避免模型"永远猜类别1"就能拿到很高的虚高准确率
def build_dataset(total_sample_num, class_num=3):
    per_class = total_sample_num // class_num
    counter = {i: 0 for i in range(class_num)}
    X, Y = [], []
    while len(Y) < per_class * class_num:
        x, y = build_sample()
        if counter[y] < per_class:
            counter[y] += 1
            X.append(x)
            Y.append(y)
    # 打乱顺序，保证每个batch里各类别都可能出现
    index = np.random.permutation(len(Y))
    X = np.array(X)[index]
    Y = np.array(Y)[index]
    return torch.FloatTensor(X), torch.LongTensor(Y)


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model, class_num=3):
    model.eval()
    test_sample_num = 300
    x, y = build_dataset(test_sample_num, class_num)
    print("本次预测集类别分布：%s" % np.bincount(y.numpy(), minlength=class_num).tolist())
    with torch.no_grad():
        y_pred = model(x)                        # (n, class_num) 每个类别的得分
        y_pred = torch.softmax(y_pred, dim=1)    # 转成概率，方便查看
        y_label = torch.argmax(y_pred, dim=1)    # 取概率最大的类别作为预测结果
        correct = int((y_label == y).sum())      # 预测类别与真实类别一致即正确
    wrong = test_sample_num - correct
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 20           # 训练轮数
    batch_size = 20          # 每次训练样本个数
    train_sample = 5000      # 每轮训练总共训练的样本总数
    input_size = 5           # 输入向量维度
    class_num = 3            # 类别数：0-负样本 1-中性样本 2-正样本
    learning_rate = 0.01     # 学习率
    hidden_size = 16         # 隐藏层大小，设为0则退化为纯线性模型
    # 建立模型
    model = TorchModel(input_size, class_num, hidden_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample, class_num)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            # 取出一个batch数据作为输入
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model(x, y)   # 计算loss  model.forward(x,y) -> cross_entropy
            loss.backward()      # 计算梯度
            optim.step()         # 更新权重
            optim.zero_grad()    # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, class_num)   # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model_new.bin")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")   # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec, class_num=3, hidden_size=16):
    input_size = 5
    model = TorchModel(input_size, class_num, hidden_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))       # (n, class_num) logits
        prob = torch.softmax(result, dim=1)                        # 转成概率
        pred_label = torch.argmax(prob, dim=1)                     # 预测类别
    class_name = {0: "负样本", 1: "中性样本", 2: "正样本"}
    for vec, res, p in zip(input_vec, pred_label, prob):
        print("输入：%s, 预测类别：%d(%s), 各类别概率：%s"
              % (vec, int(res), class_name[int(res)], [round(float(i), 4) for i in p]))


if __name__ == "__main__":
    main()
    test_vec = [
        [0.81889086, 0.15229675, 0.31082123, 0.03504317, 0.88920843],  # 最大值在第5维 -> 正样本
        [0.90797868, 0.67482528, 0.13625847, 0.34675372, 0.19871392],  # 最大在第1维、最小在第3维 -> 中性
        [0.94963533, 0.55242560, 0.95758807, 0.95520434, 0.84890681],  # 最大在第3维 -> 中性
        [0.03504317, 0.15229675, 0.31082123, 0.81889086, 0.08920843],  # 最小值在第1维 -> 负样本
    ]
    predict("model_new.bin", test_vec)
