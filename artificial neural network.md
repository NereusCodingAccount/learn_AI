# 神經網路架構速覽

#### 主要元件（簡潔）####

## 層（Layer）：Dense/Linear、Conv、RNN、Self-Attention。

# 層（Layer）是神經網路中的基本運算單元，用來把輸入轉換成輸出，並逐層抽取或組合特徵。簡要說明：

# 基本數學形式

許多層可表示為 y = f(Wx + b)（可訓練參數 W, b；f 為啟動函數）。
有些層是無參數的（例如 MaxPool、Flatten、Activation 若單獨算作層）。
常見層類型（用途與特性）

- **Dense / Fully‑connected**：每個輸入連到每個輸出；適合表格資料、分類/回歸。

- **Convolution（Conv）**：用卷積核在局部窗口計算，保留空間結構，常用於影像。

- **Recurrent（RNN / LSTM / GRU）**：處理序列，保留時間狀態（hidden state）。

- **Attention / Self‑Attention**：以注意力機制建模序列中元素間的關聯（Transformer）。

- **Normalization（BatchNorm / LayerNorm）**：標準化激活值，穩定訓練、加速收斂（有時有可學參數）。

- **Pooling（MaxPool / AvgPool）**：空間下採樣，減少尺寸與平移不變性。

- **Dropout**：訓練時隨機丟棄神經元，做正則化（無參數）。

## 輸入/輸出形狀（shape）觀念

**Dense**：輸入 (batch, in_dim) → 輸出 (batch, out_dim)。
**Conv2D**：輸入 (batch, C, H, W) → 輸出 (batch, C_out, H', W')（依 kernel、stride、padding 決定 H', W'）。
**RNN**：輸入 (seq_len, batch, feature) 或 (batch, seq_len, feature) → 輸出含時間維度。
層的角色與設計要點

每層學到不同層次的表示（淺層學局部/低階特徵，深層學抽象/高階特徵）。
層的順序與種類決定網路能力與計算量（例如 Conv→Pool→Conv→Flatten→Dense）。
小心參數數量（過多會過擬合、過少表達力不足）；搭配正則化與適當深度。  

# 啟動函數：ReLU、Sigmoid、Tanh、GELU。  

# 損失函數：分類→Cross-Entropy，回歸→MSE。  
# 優化器：SGD、Adam（常用）。  
# 正則化：Dropout、BatchNorm/LayerNorm、L2（權重衰減）。  
# 評估指標：Accuracy、F1、ROC-AUC、MAE/MSE。

## 常見架構（用途與重點）
- MLP（多層感知器）：表格資料、簡單分類/回歸。  
- CNN（卷積網路）：影像、局部特徵擷取、池化與卷積核設計很重要。  
- RNN / LSTM / GRU：序列資料（文字、時間序列）。  
- Transformer（自注意力）：NLP 與序列建模，具平行化與長距離依賴處理能力。  
- Autoencoder：降維、去噪、表示學習。  
- GAN：生成模型，訓練需注意不穩定性與模式崩潰。  
- GNN：處理圖結構資料（消息傳遞）。

## 設計與調參要點（實務重點）
- 先從小模型與少量層開始，逐步擴大。  
- 學習率最敏感：使用 lr 調度、warmup。  
- 使用 BatchNorm/LayerNorm 幫助穩定與加速收斂。  
- 避免過擬合：資料擴增、Dropout、早停（early stopping）。  
- batch size 與 lr 相關，可用線性縮放原則調整。  
- 分訓練/驗證/測試集，確保評估可靠性。

## 簡短範例（PyTorch）
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden=128, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(64*8*8, num_classes)  # 假設輸入 32x32
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)
```

## Transformer Encoder（簡骨架）
```python
import torch.nn as nn

class SimpleTransformerEncoder(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    def forward(self, src, src_mask=None):
        # src shape: (seq_len, batch, d_model)
        return self.encoder(src, mask=src_mask)
```

---

需要我把某一部分（例如完整訓練迴圈、資料載入或 Keras 範例）補到檔案裡嗎？