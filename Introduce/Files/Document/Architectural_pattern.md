# Architectural Patterns 架構模式

## 由AI生成的文檔:
#### 指 「神經網路的結構設計模式（architectural patterns）」，也就是「整個網路是怎麼連接不同層、傳遞訊息」的設計思路。

#### 它們不是單一層（Layer），而是網路結構的組合方式或連接策略。

---

### 殘差（Residual） / 跳接（Skip Connection）

#### 代表模型：ResNet

### 概念：
#### 在深層網路中，訊息會越傳越模糊、梯度會消失。

#### 「殘差連接」讓輸入的訊號直接跳過中間幾層，再與輸出相加，幫助訊息保留。

![alt text](image.png)

---

## 多分支（Inception 類型）

#### 代表模型：GoogLeNet（Inception 系列）

### 概念：
#### 不同卷積核（3×3、5×5、1×1）可以觀察不同「尺度」的特徵。
#### Inception 模組同時用多種卷積核，再把結果**拼接（concatenate）**起來。

### 好處：

- #### 模型在同一層學習多種「特徵視角」。
- #### 可並行運算，加快訓練速度。

---

## 編碼器-解碼器（Encoder-Decoder）

### 代表模型：U-Net、Transformer、Autoencoder

### 概念：

#### **Encoder（編碼器）**：把輸入壓縮成特徵向量（抽象表示）
#### **Decoder（解碼器）**：再從這些特徵恢復輸出（重建或生成）

### 用途：

- #### 自動編碼器（Autoencoder）：影像壓縮 / 去雜訊
- #### U-Net：影像分割（Encoder 壓縮 → Decoder 還原）
- #### Transformer：語言翻譯（Encoder 處理輸入句子 → Decoder 生成輸出句子）

---
## References 參考

1. https://eprints.cs.univie.ac.at/2698/1/ArchPatterns.pdf

# [返回](../../ANN.md)