# Optimizer（參數更新方式）介紹

## "目前找不到資料，所以先用AI寫的"

#### Optimizer（優化器）負責根據損失函數的梯度更新模型參數，以最小化損失。核心概念是把計算得來的梯度轉換成參數的調整量。常見類別與重點如下。

---

## 常見 Optimizer 與特性

### SGD（Stochastic Gradient Descent）  
  - 更新：theta ← theta − lr * grad  
  - 優點：簡單、泛化性佳（搭配適當 lr 與 momentum）。  
  - 通常與 momentum 一起使用。

### Momentum（加動量的 SGD）  
  - 保留過去更新的指向以加速收斂並減少震盪：  
    v ← mu * v + lr * grad；theta ← theta − v

### Nesterov Momentum（Nesterov Accelerated Gradient）  
  - 預先朝動量方向看一小步再計算梯度，能更早修正更新方向。

### AdaGrad  
  - 對每個參數記錄平方梯度累積，適合稀疏特徵，但累積會導致學習率快速衰減。

### RMSprop  
  - 用指數移動平均（EMA）追蹤平方梯度，避免 AdaGrad 的學習率衰退過快。

### Adam（Adaptive Moment Estimation）  
  - 結合 momentum（m）與 RMSprop（v），加上偏差校正，是現代常用的 optim。  
  - 預設適用於很多任務（lr=1e-3 常見）。

### AdamW  
  - 將 weight decay（權重衰減）從梯度更新中解耦，對正規化更直觀，Transformer 類模型常用。

### 其他：AdaMax、Nadam、RAdam、Lookahead 等，皆為不同改善或穩定化策略。

---

## 正則化與技巧
- Weight decay（L2）與 AdamW：用來防止過擬合；AdamW 建議使用。  
- Gradient clipping：clip_grad_norm_ 或 clip_grad_value_，防止梯度爆炸（RNN 或大 lr 時常用）。  
- Learning rate schedule：step decay、cosine annealing、exponential、warmup（先小 lr 漸增）等，可顯著提升訓練穩定性與最終效果。  
- LR 找尋器（LR finder）：快速找到合理 lr 範圍（如 fastai 方法）。

---

## 實務建議（簡短）
- lr 是最敏感的超參數；先試 lr 探索（例如 1e-1 → 1e-6 範圍）。  
- 小模型或小資料集可從 Adam（lr=1e-3）開始；大型影像訓練多用 SGD + momentum（lr 0.01–0.1，視 batch 而定）。  
- 使用 weight decay 約 1e-4 ~ 1e-2（依模型與任務調整）。  
- 加入 scheduler 與 warmup 能顯著提升大型模型的訓練穩定性。  
- 定期 checkpoint，並在訓練中加入 gradient clipping 以避免不穩定。


# [返回](../../ANN.md)