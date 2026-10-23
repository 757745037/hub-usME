# 多分类机器学习任务
# 任意的五维向量，最大值在第几维就属于第几类

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt


class TorchModel(nn.Module):
    def __init__(self, in_features):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(in_features, 5)
        self.loss = nn.CrossEntropyLoss()
        
    def forward(self, x):
        return self.linear(x)  # raw logits; softmax is applied inside CrossEntropyLoss
        


def evaluate(model: nn.Module, x, y):
    model.eval()

    print("本次预测集中样本分布: ", y.sum(dim=0))
    
    correct, wrong = 0, 0
    with torch.no_grad(): # predication
        y_pred = torch.softmax(model(x), dim=1)  # probabilities for prediction
        for y_p, y_t in zip(y_pred, y):
            if np.argmax(y_p) == np.argmax(y_t):
                correct += 1
            else:
                wrong += 1
    accurate = correct / (correct + wrong)
    print("正确预测样本个数: %d, 正确率: %f" % (correct, accurate))
    return accurate
            

def to_one_hot(target, shape):
    one_hot_target = np.zeros(shape)
    for i, t in enumerate(target):
        index = int(t.item())
        one_hot_target[i][index] = 1
    return one_hot_target


# 生成一个样本
def build_sample():
    x = np.random.random(5)
    y = np.argmax(x, axis=0)
    return x, y


def build_dataset(num):
    X = np.zeros((num, 5))
    Y = np.zeros((num, 1))
    
    for i in range(num):
        x, y = build_sample()
        X[i] = x
        Y[i] = y
    
    Y = to_one_hot(target=Y, shape=(num, 5))
    
    return torch.FloatTensor(X), torch.FloatTensor(Y)


def main():
    epochs = 30
    batch_size = 30
    train_smaple = 1000
    input_size = 5
    lr = 0.01
    
    model = TorchModel(in_features=input_size)
    print("initial weight: %s" % model.state_dict())
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    log = []

    train_x, train_y = build_dataset(train_smaple)
    test_x, test_y = build_dataset(1000)  # fixed test set, reused every epoch
    # training process
    for epoch in range(epochs):
        model.train()
        watch_loss = []
        for batch in range(train_smaple // batch_size):
            x = train_x[batch_size*batch:(batch+1)*batch_size]
            y = train_y[batch_size*batch:(batch+1)*batch_size]
            y_pred = model(x)
            loss = model.loss(y_pred, y)
            loss.backward() # 计算梯度
            optim.step() # 更新权重
            optim.zero_grad()
            watch_loss.append(loss.item())
        # print("=============\n epoch=%d 平均loss: %f" % (epoch, np.mean(watch_loss)))
        acc = evaluate(model, test_x, test_y)
        log.append([acc, float(np.mean(watch_loss))])
    
    torch.save(model.state_dict(), "model.bin")
    print(model.state_dict())
        
    # 画图
    # print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.savefig("loss.png")


def predict(model_path):
    test_data = build_dataset(5)
    
    model = TorchModel(5)
    model.load_state_dict(torch.load(model_path))

    model.eval()
    with torch.no_grad():        
        result = torch.softmax(model(test_data[0]), dim=1)
    
    for vec, res in zip(test_data[0], result):
        print("输入: %s, 预测类别: %d, 概率值: %f" % (vec, np.argmax(res), res[np.argmax(res)]))
    
    

if __name__ == '__main__':
    main()
    predict("model.bin")