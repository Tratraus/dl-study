import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

# 数据创建
X = torch.randn(10, 2)
Y = (X[:, 0] * X[:, 1] > 0).long()

# Dataset 和 DataLoader
dataset = torch.utils.data.TensorDataset(X, Y)
loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

# 模型定义
model = nn.Sequential(
  nn.Linear(2, 4),
  nn.ReLU(),
  nn.Linear(4, 2)
)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)



# 训练循环
for epoch in range(100):
    for batch_X, batch_Y in loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_Y)
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# 测试模型
with torch.no_grad():
    logits = model(X)
    predictions = torch.argmax(logits, dim=1)
    accuracy = (predictions == Y).float().mean()
    print(f'Accuracy: {accuracy.item() * 100:.2f}%')

# 输出问题 1
# 如果今天让你不看笔记，再写一遍最小训练循环，你觉得自己最容易忘的是哪一步？为什么？
# 主要是一些函数的写法，目前对于我应该选择什么函数及其写法还是不太熟练


# 输出问题 2
# 现在请你用一句话概括：Dataset、DataLoader、model、loss、backward、optimizer.step 各自的作用。
# Dataset：封装数据
# DataLoader：批量加载数据
# model：定义模型结构
# loss：计算损失
# backward：计算梯度
# optimizer.step：更新模型参数

# 输出问题 3
# 为什么分类任务里标签通常是 (N,)，而不是 (N,2)？
# 不太确定
# Answer:
# 因为 CrossEntropyLoss 期望标签是类别编号，不是 one-hot 向量。
# 对于二分类任务，如果有 N 个样本，标签通常写成：
# y = tensor([0, 1, 0, 1, 1])
# shape 是 (N,)，每个元素是类别编号（0 或 1）。
# 如果写成 (N, 2)，一般是 one-hot 标签，比如：
# tensor([
#     [1, 0],
#     [0, 1],
#     [0, 1],
#     [1, 0]
# ])
# 这也是一种表达分类标签的方法，但它不是 nn.CrossEntropyLoss() 默认需要的格式。


# 输出问题 4
# 经过这一周后，你觉得自己对“模型训练”这件事的理解发生了什么变化？
# 首先是对模型训练的结构有了更深的认知，其次是了解了一些相关的函数