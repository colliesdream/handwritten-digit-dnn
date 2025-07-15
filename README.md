# 🧠 Handwritten Digit Recognition (NumPy DNN + GUI)

這是一個基於 **NumPy 手刻深度神經網路（DNN）** 的手寫數字辨識系統，搭配簡易 GUI 輸入介面，能即時辨識 0～9 的手寫數字。

資料集採用 [MNIST](http://yann.lecun.com/exdb/mnist/)，本專案完全不依賴 TensorFlow/PyTorch，適合作為 DNN 原理教學與實作範例。

---

## 📁 專案結構

```bash
.
├── dnn_model.npz         # 訓練好的模型參數（W1~W3, b1~b3）
├── train_model.py        # DNN 模型訓練腳本（手刻 forward/backward）
├── predict_gui.py        # GUI 輸入互動 + 即時推論
├── loss_curve.png        # 訓練過程中的 Loss 曲線圖
├── requirements.txt      # 套件需求列表
└── README.md             # 專案說明
```

---

## 🚀 快速開始

### 1. 安裝相依套件（建議使用虛擬環境）

```bash
pip install -r requirements.txt
```

### 2. 啟動手寫輸入 GUI

```bash
python predict_gui.py
```

- 用滑鼠左鍵在白色畫布上手寫數字
- 點選「辨識」按鈕後，模型即時輸出結果

### 3. 若要重新訓練模型

```bash
python train_model.py
```

- 訓練完會自動儲存 `dnn_model.npz`
- 並繪製 `loss_curve.png` 供觀察學習趨勢

---

## 🧠 模型架構

| 層級       | 輸入 | 輸出 | Activation |
|------------|------|------|------------|
| Input      | 784  | -    | -          |
| Hidden 1   | 784  | 128  | ReLU       |
| Hidden 2   | 128  | 64   | ReLU       |
| Output     | 64   | 10   | Softmax    |

- 使用交叉熵作為損失函數（Cross-Entropy Loss）
- Optimizer 為單純的 SGD

---

## ✨ GUI 預處理邏輯（get_canvas_image）

為了讓手寫輸入與 MNIST 格式一致，會進行：

1. 將白底黑字轉為黑底白字（反相）
2. 裁切手寫區域 bounding box
3. 中心對齊置中
4. resize 成 28x28
5. 正規化（除以 255）

---

## 📸 預期截圖

> 可加入執行 predict_gui.py 的操作畫面做展示

---

## 🔧 待改進項目

- ✅ 圖像預處理裁切與中心化修正
- 🔲 GUI 支援更流暢書寫與刪除
- 🔲 模型封裝（ModelLoader / Inference API）
- 🔲 轉為 CNN 架構進一步提升辨識率
- 🔲 將 GUI 部署成 Web 版介面（如 Flask）

---

## 📄 License

本專案採用 MIT License，可自由使用與修改。

---

## 🙌 Credit

資料來源：[MNIST 手寫數字集](http://yann.lecun.com/exdb/mnist/)
