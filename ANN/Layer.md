# 層（Layer）

## 人工智能（AI）神經網絡中的每個層（Layer）功能不同，下面介紹常見的幾種層及其作用：

### 輸出層 (Input Layer):
  - **功能**：接收原始數據（如圖像像素、語音信號或文本特徵）。
  - **作用**：將外部數據轉換為網絡可以處理的形式，不進行計算，只是數據的入口。

### 隱藏層 (Hidden Layers):
  - **功能**：介於輸入層和輸出層之間，負責處理和轉換數據。
  - **作用**：從數據中抽取特徵，進行非線性映射，學習複雜模式。
  - **特殊層**

<details>
<summary>隱藏層的詳細介紹</summary>

# Dence / Fully‑connected(全連接層)：
- 最基本的層，把上一層的所有輸入節點都連接到本層的每個神經元；適合**表格資料、分類/回歸**。
- 它會對輸入資料進行加權線性組合，再加上一個偏差項（bias），公式如下：
### y(輸出) = W(權重矩陣)∙x(輸入) + b(偏差向)
- **特點**:有更高的靈活度和學習能力，通常需要硬體(ex:GPU)來提升運算速度。

```python
from tensorflow.keras.layers import Dense

layer = Dense(128, activation='relu')# 建立128個神經元的全連接層

```
---
# Convolution / Conv(卷積層)
- 深度學習中**圖像處理**任務的重要核心。

- 卷積層的主要作用是利用「卷積核」或「濾波器」在輸入資料（如影像）上滑動，提取區域性特徵，例如邊緣、顏色或紋理。每個卷積核會學習一組權重，能針對不同圖像特徵有不同的敏感度。

- 卷積層的關鍵參數:
  - kernel size（卷積核大小）
  - stride（步長）
  - padding（填補法，有「valid」與「same」兩種）
  - filter/深度數量

```python
from tensorflow.keras.layers import Conv2D

layer = Conv2D(filters=32, kernel_size=(3,3), activation='relu')
# 產生 32 張特徵圖，每個卷積核大小 3×3
```
---

</details>

### 輸出層 (Output Layer):
  - **功能**：輸出最終預測結果。
  - **作用**：根據任務（分類、回歸等）生成結果，如分類概率或回歸值。
  - 節點數量通常和輸出類別數對應。


##