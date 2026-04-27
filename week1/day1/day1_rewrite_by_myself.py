import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

# create data
x = torch.randn(200, 4)
y = (x[:,0] + x[:,1] > x[:,2] + x[:,3]).long()

# use very small train set
x_train = x[:10]
y_train = y[:10]

x_test = x[20:]
y_test = y[20:]

# stronger model
model = nn.Sequential(
    nn.Linear(4, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 2)
)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(1000):
    # train
    model.train()
    logits = model(x_train)
    loss = criterion(logits, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        # eval
        model.eval()
        with torch.no_grad():
            train_logits = model(x_train)
            train_pred = train_logits.argmax(dim=1)
            train_acc = (train_pred == y_train).float().mean()

            test_logits = model(x_test)
            test_pred = test_logits.argmax(dim=1)
            test_acc = (test_pred == y_test).float().mean()

        print(
            f"Epoch {epoch+1:04d} | "
            f"Loss = {loss.item():.4f} | "
            f"Train Acc = {train_acc.item():.4f} | "
            f"Test Acc = {test_acc.item():.4f}"
        )
