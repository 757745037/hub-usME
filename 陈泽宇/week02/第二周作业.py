"""第二次作业
任务：输入一个五维向量，最大元素出现在第几维，就属于第几类。
程序内部使用 0~4 作为标签；输出给人看时转换为 1~5 类。
"""

# %% 1. 导入库与设置随机种子
from pathlib import Path
import json
import random

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 40
BATCH_SIZE = 256
LEARNING_RATE = 0.005
OUTPUT_DIR = Path(__file__).resolve().parent / 'outputs'
OUTPUT_DIR.mkdir(exist_ok=True)
print('使用设备:', DEVICE)

# %% 2. 人工生成数据：五维向量 -> 最大值所在维度
# 使用连续分布抽样，几乎不会有并列最大值。
# np.argmax(x, axis=1) 返回 0、1、2、3、4，分别对应第 1~5 类。
def make_dataset(size, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(loc=0.0, scale=1.0, size=(size, 5)).astype(np.float32)
    y = np.argmax(x, axis=1).astype(np.int64)
    return torch.from_numpy(x), torch.from_numpy(y)

x_train, y_train = make_dataset(10000, SEED)
x_val, y_val = make_dataset(2000, SEED + 1)
x_test, y_test = make_dataset(2000, SEED + 2)
train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=BATCH_SIZE,
                          shuffle=True, generator=torch.Generator().manual_seed(SEED))
print('训练/验证/测试样本数:', len(x_train), len(x_val), len(x_test))
print('第一个输入:', x_train[0].tolist(), '正确类别:', y_train[0].item() + 1)

# %% 3. 构建多层感知机：5 -> 32 -> 16 -> 5
class FiveClassNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 5),
        )

    def forward(self, x):
        # 逐样本标准化：减去五维均值，再除以标准差。
        # 正数倍缩放或整体平移不改变最大值的位置，因此这样可适应不同数值范围。
        x = (x - x.mean(dim=1, keepdim=True)) / (
            x.std(dim=1, keepdim=True, unbiased=False) + 1e-6
        )
        return self.network(x)  # 输出 5 个 logits；不需要提前做 softmax

model = FiveClassNet().to(DEVICE)
criterion = nn.CrossEntropyLoss()  # 已内置 LogSoftmax + 负对数似然
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# %% 4. 训练并监控准确率
@torch.no_grad()
def evaluate(x, y):
    model.eval()
    logits = model(x.to(DEVICE))
    loss = criterion(logits, y.to(DEVICE)).item()
    pred = logits.argmax(dim=1).cpu()
    accuracy = (pred == y).float().mean().item()
    return loss, accuracy, pred

history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
for epoch in range(1, EPOCHS + 1):
    model.train()
    for bx, by in train_loader:
        bx, by = bx.to(DEVICE), by.to(DEVICE)
        optimizer.zero_grad()
        logits = model(bx)
        loss = criterion(logits, by)
        loss.backward()            # 反向传播
        optimizer.step()           # 更新模型参数

    train_loss, train_acc, _ = evaluate(x_train, y_train)
    val_loss, val_acc, _ = evaluate(x_val, y_val)
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
        print(f'Epoch {epoch:02d}/{EPOCHS} | train loss={train_loss:.4f}, '
              f'train acc={train_acc:.2%} | val loss={val_loss:.4f}, val acc={val_acc:.2%}')

# %% 5. 测试集评估、混淆矩阵与规则方法对照

test_loss, test_acc, test_pred = evaluate(x_test, y_test)
print('\n测试集结果: loss={:.4f}, accuracy={:.2%}'.format(test_loss, test_acc))
confusion = np.zeros((5, 5), dtype=int)
for actual, predicted in zip(y_test.numpy(), test_pred.numpy()):
    confusion[actual, predicted] += 1
print('混淆矩阵（行=实际类别，列=预测类别）:\n', confusion)

rule_acc = np.mean(np.argmax(x_test.numpy(), axis=1) == y_test.numpy())
print('直接使用 argmax 规则的准确率: {:.2%}'.format(rule_acc))

# %% 6. 画训练曲线和混淆矩阵
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(history['train_loss'], label='Training')
axes[0].plot(history['val_loss'], label='Validation')
axes[0].set(xlabel='Epoch', ylabel='Cross-entropy loss', title='Loss')
axes[0].grid(alpha=0.25)
axes[0].legend()
axes[1].plot(np.array(history['train_acc']) * 100, label='Training')
axes[1].plot(np.array(history['val_acc']) * 100, label='Validation')
axes[1].set(xlabel='Epoch', ylabel='Accuracy (%)', title='Accuracy', ylim=(0, 102))
axes[1].grid(alpha=0.25)
axes[1].legend()
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'training_curves.png', dpi=180)
plt.close(fig)

fig, ax = plt.subplots(figsize=(5.6, 4.8))
img = ax.imshow(confusion, cmap='Blues')
fig.colorbar(img, ax=ax)
ax.set_xticks(range(5), labels=[f'C{i}' for i in range(1, 6)])
ax.set_yticks(range(5), labels=[f'C{i}' for i in range(1, 6)])
ax.set(xlabel='Predicted class', ylabel='True class', title='Test confusion matrix')
for i in range(5):
    for j in range(5):
        ax.text(j, i, str(confusion[i, j]), ha='center', va='center',
                color='white' if confusion[i, j] > confusion.max() / 2 else 'black')
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'confusion_matrix.png', dpi=180)
plt.close(fig)

# %% 7. 输入新向量并预测
@torch.no_grad()
def predict_one(vector):
    if len(vector) != 5:
        raise ValueError('输入必须恰好包含 5 个数字')
    model.eval()
    xx = torch.tensor([vector], dtype=torch.float32, device=DEVICE)
    logits = model(xx)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return int(np.argmax(probs)) + 1, probs

examples = [
    [0.2, 0.8, -0.1, 0.3, 0.5],    # 第 2 维最大
    [-3.0, -2.0, -1.0, 4.0, 0.0],  # 第 4 维最大
    [500.0, 1000.0, -20.0, 3.0, 8.0],  # 测试较大数值范围
]
for v in examples:
    pred_class, probs = predict_one(v)
    print(f'输入: {v} | 网络预测: 第{pred_class}类 | 规则答案: 第{int(np.argmax(v))+1}类 | '
          f'预测概率: {np.round(probs, 3)}')

# %% 8. 保存模型及实验指标

torch.save(model.state_dict(), OUTPUT_DIR / 'five_class_model.pth')
metrics = {
    'seed': SEED,
    'epochs': EPOCHS,
    'train_size': len(x_train),
    'val_size': len(x_val),
    'test_size': len(x_test),
    'test_loss': round(test_loss, 6),
    'test_accuracy': round(test_acc, 6),
    'rule_accuracy': float(rule_acc),
    'confusion_matrix': confusion.tolist(),
    'sample_predictions': [
        {'input': v, 'predicted_class': predict_one(v)[0], 'rule_class': int(np.argmax(v)) + 1}
        for v in examples
    ],
}
(OUTPUT_DIR / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
print('\n结果文件已保存至:', OUTPUT_DIR)

