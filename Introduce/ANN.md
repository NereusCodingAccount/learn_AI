# 神經網路架構（簡要說明）

神經網路架構（Neural Network Architecture）又名**類神經網路**是指網路中各層的類型、順序與連接方式，以及輸入/輸出的形狀與資料流動規則。它決定模型的表示能力、計算成本與適用場景。

## 組成要素（重點）
- 前往[Layer(層)](./Files/Document/Layer.md) 
- [啟動函數](./Files/Document/Activation_function.md)（Activation）：非線性變換，例如 ReLU、Sigmoid、Tanh、GELU。  
- [損失函數](./Files/Document/Loss_function.md)（Loss）：評估模型輸出與目標差距（分類常用 Cross‑Entropy，回歸用 MSE）。  
- 優化器（Optimizer）：更新參數的方法，如 SGD、Adam。  
- 結構模式：殘差（Residual）/ 跳接（Skip）、多分支（Inception 類型）、編碼器‑解碼器（Encoder‑Decoder）等。

## 常見架構類型（用途）
- MLP（多層感知器）：表格資料、簡單分類/回歸。  
- 前往[CNN](./Files/Document/CNN.md)（卷積神經網路）：影像、視覺特徵提取。  
- 前往[RNN](./Files/Document/RNN.md) / 前往[LSTM](./Files/Document/LSTM.md) / 前往[GRU](./Files/Document/GRU.md)：序列資料（文字、時間序列、語音）。  
- Transformer（Self‑Attention 為核心）：NLP、序列建模與大型預訓練模型。  
- Autoencoder / GAN / GNN：降維、生成式建模、圖資料處理。

## 設計考量（實務要點）
- 深度（層數）vs 寬度（每層神經元數）：影響表現與過擬合風險。  
- 感受野（Receptive Field）：尤其在 CNN 中決定能抓多大範圍的特徵。  
- 正則化：Dropout、權重衰減、資料擴增、早停等避免過擬合。  
- 正規化層：BatchNorm / LayerNorm 幫助穩定與加速訓練。  
- 訓練細節：學習率與其排程（scheduler）、批次大小、初始化方法對收斂影響最大。  
- 模組化：把常用塊（conv‑bn‑relu、residual block）封裝以利復用與測試。

## 快速流程（從架構到部署）
1. 根據資料與任務選擇基本架構（例如影像選 CNN）。  
2. 設計層序列、尺寸與參數（kernel、stride、units）。  
3. 設定損失、優化器與正則化策略。  
4. 訓練、驗證並調整超參數（lr、batch size、層數）。  
5. 儲存 checkpoint，轉為推理/部署（注意切換 model.eval()）。

## Reference
1. https://en.wikipedia.org/wiki/Neural_network_(machine_learning)