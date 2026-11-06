# coding:utf8

# 解决 OpenMP 库冲突问题
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：每个数字根据自己在 5 个数字里的排名，对应自己的类别（第 1 名～第 5 名），5 个数字各有各的类别。

"""

class TorchModel(nn.Module):
    def __init__(self, input_size=5, num_classes=5, hidden_size=128):
        super(TorchModel, self).__init__()
        # 第一层：升维提取非线性特征
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()  # 非线性激活，最核心的改动
        # 第二层：输出每个位置的分类得分
        self.fc2 = nn.Linear(hidden_size, num_classes * num_classes)
        self.loss = nn.CrossEntropyLoss()
        self.num_classes = num_classes

    def forward(self, x, y=None):
        # 两层网络 + 非线性激活
        x = self.fc1(x)
        x = self.relu(x)
        logits = self.fc2(x)
        
        if y is not None:
            logits = logits.view(-1, self.num_classes)
            y_flat = y.view(-1)
            return self.loss(logits, y_flat)
        else:
            logits = logits.view(-1, 5, self.num_classes)
            return torch.softmax(logits, dim=2)



# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，如果第一个值大于第五个值，认为是正样本，反之为负样本
def build_sample():
    x = np.random.random(5)
    ranks = np.argsort(np.argsort(-x))
    return x, ranks


# 随机生成一批样本
# 正负样本均匀生成
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    # 先转成numpy数组再转tensor，消除警告；标签转LongTensor
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.int64)
    # print(X)
    # print(Y)
    return torch.from_numpy(X), torch.from_numpy(Y)

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    correct = 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测 model.forward(x)
        pred_ranks = torch.argmax(y_pred, dim=2)
        correct = (y == pred_ranks).sum().item()
    total = test_sample_num * 5  # 总共有100×5个位置需要预测
    acc = correct / total
    print(f"总位置数：{total}，正确预测位置数：{correct}，位置准确率：{acc:.4f}")
    return acc

def main():
    # 配置参数
    epoch_num = 200  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 1000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.005  # 学习率
    # 建立模型
    model = TorchModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size): 
            #取出一个batch数据作为输入   train_x[0:20]  train_y[0:20] train_x[20:40]  train_y[20:40]
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss  model.forward(x,y)
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model_rank.bin")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    model = TorchModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    with torch.no_grad():
        result = model(torch.FloatTensor(np.array(input_vec)))
    
    for vec, res in zip(input_vec, result):
        pred_ranks = torch.argmax(res, dim=1).numpy()
        print(f"输入向量：{np.round(vec, 4)}")
        print(f"预测排名（0=最大）：{pred_ranks}")
        print(f"真实排名：{np.argsort(np.argsort(-np.array(vec)))}")
        print("-"*50)


if __name__ == "__main__":
    main()
    test_vec = [
        [0.8189, 0.1523, 0.3108, 0.0350, 0.8892],
        [0.9496, 0.5524, 0.9576, 0.9552, 0.8489]
    ]
    predict("model_rank.bin", test_vec)

