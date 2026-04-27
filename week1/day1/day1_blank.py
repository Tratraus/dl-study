import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

X = torch.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).long()

model = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 2)
)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(20):
    logits = model(X)
    loss = criterion(logits, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 5 == 0:
        pred = logits.argmax(dim=1)
        acc = (pred == y).float().mean()
        print(f"Epoch {epoch+1:02d} | Loss = {loss.item():.4f} | Acc = {acc.item():.4f}")
