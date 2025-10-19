import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# 簡單 MLP 模型
class SimpleMLP(nn.Module):
    def __init__(self, input_dim=2, hidden=32, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes)
        )
    def forward(self, x):
        return self.net(x)

# 產生合成二分類資料（兩個高斯群）
def make_blobs(n_samples=1000):
    n1 = n_samples // 2
    n2 = n_samples - n1
    # 群 1: centered at (-1, 0), 群 2: centered at (1, 0)
    x1 = torch.randn(n1, 2) * 0.5 + torch.tensor([-1.0, 0.0])
    x2 = torch.randn(n2, 2) * 0.5 + torch.tensor([1.0, 0.0])
    X = torch.cat([x1, x2], dim=0)
    y = torch.cat([torch.zeros(n1, dtype=torch.long), torch.ones(n2, dtype=torch.long)], dim=0)
    # 隨機打亂
    perm = torch.randperm(n_samples)
    return X[perm], y[perm]

def train():
    device = torch.device("cpu")
    X, y = make_blobs(1000)
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256)

    model = SimpleMLP(input_dim=2, hidden=32, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 50
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        # 驗證
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == yb).sum().item()
                val_total += xb.size(0)
        val_loss /= val_total
        val_acc = val_correct / val_total

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}  train_loss={train_loss:.4f} train_acc={train_acc:.4f}  val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    # 儲存模型
    torch.save(model.state_dict(), "simple_mlp.pth")
    print("訓練完成，模型已儲存為 simple_mlp.pth")

if __name__ == "__main__":
    train()