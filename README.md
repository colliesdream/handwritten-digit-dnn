# 手寫數字辨識系統（NumPy 手刻 DNN + Tkinter GUI）

本專案是一個簡易但完整的 **手寫數字辨識系統**，包含：

- 使用 NumPy 手刻多層感知機（DNN）訓練模型
- 利用 `Tkinter` 製作互動式手寫畫布與推論介面
- 完整訓練、預測流程（含模型儲存、載入與可視化）
- 前處理包含裁切、反相、縮放、置中、模糊與正規化

---

## 📌 系統架構

- **輸入介面**：Tkinter 畫布（200×200 手寫區）
- **模型架構**：784 → 128 → 64 → 10，全使用 ReLU + Softmax
- **訓練資料**：使用 [MNIST 手寫數字資料集](http://yann.lecun.com/exdb/mnist/)
- **初始化方式**：使用 He initialization，提升深層網路收斂效果

---

## 🔧 使用技術

| 類別       | 技術 / 方法                     |
|------------|----------------------------------|
| 前處理     | PIL、ImageOps、Gaussian Filter、置中裁切等 |
| 訓練       | NumPy 手刻前向 / 反向傳播，Cross Entropy Loss，Mini-Batch 梯度下降 |
| 初始化     | He initialization               |
| 資料集     | MNIST（由 TensorFlow 載入）       |
| GUI        | Tkinter + PIL                   |
| 模型儲存   | `np.savez` 儲存六組權重參數        |

---

## 🚀 如何使用

### 1️⃣ 安裝必要套件
```bash
pip install numpy matplotlib pillow pyscreenshot tensorflow
```

### 2️⃣ 訓練模型
```bash
python train_dnn.py
```
> 會訓練 100 epochs 並自動儲存 `dnn_model.npz`

### 3️⃣ 啟動手寫辨識 GUI
```bash
python predict_gui.py
```

### 4️⃣ 操作方式
- 在畫布上手寫數字
- 點擊「辨識」顯示預測結果與 Top-3 機率
- 點擊「預覽影像」可檢視模型實際收到的 28×28 圖像

---

## 學習心得

這個專案是我在學習 AI 與深度學習過程中，結合數學基礎與程式實作的一次寶貴學習經驗，雖然程式碼由ai協助生成，但我從一個完整的可運行的小辨識模型中學到以下理論在實作中的角色。在 AI 協助與查詢的引導下，我逐步理解了：

- DNN 的前向 / 反向傳播過程
- Softmax 與 Cross Entropy 的意義與實作
- 初始化方法對訓練成效的影響（特別是 He initialization）
- 前處理對辨識率的關鍵角色（裁切、置中、反相等）

雖然 GUI 與前處理整合上由 AI 提供許多建議，但透過持續 debug 與觀察辨識效果，我已能掌握整體模型的運作邏輯。

---

## 📁 結構

```
│
├── train_dnn.py         # 模型訓練程式碼
├── predict_gui.py       # GUI 與模型推論程式
├── dnn_model.npz        # 儲存的模型權重
├── demo/
│   ├── gui_demo.png     # 介面展示截圖
│   └── loss_curve.png   # Loss 曲線圖
```

