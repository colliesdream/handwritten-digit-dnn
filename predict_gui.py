import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import pyscreenshot as ImageGrab
import os
from scipy.ndimage import gaussian_filter

# 載入手刻模型參數
data = np.load("dnn_model.npz")
W1, b1 = data['W1'], data['b1']
W2, b2 = data['W2'], data['b2']
W3, b3 = data['W3'], data['b3']

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)

def predict(x):
    z1 = x.dot(W1) + b1
    a1 = relu(z1)
    z2 = a1.dot(W2) + b2
    a2 = relu(z2)
    z3 = a2.dot(W3) + b3
    return softmax(z3)

def get_canvas_image():
    # 將 canvas 儲存為 postscript 再轉 PIL
    canvas.postscript(file="temp_canvas.eps", colormode='color')
    img = Image.open("temp_canvas.eps").convert('L')  # 強制轉灰階

    # ✅ 反相（因為是白底黑字）
    img = ImageOps.invert(img)

    # ✅ 裁切非白背景區域
    bbox = img.getbbox()
    if bbox is None:
        print("⚠️ 沒有有效繪圖內容")
        return np.zeros((1, 784))
    img = img.crop(bbox)

    # ✅ 縮放最大邊為 20，等比例
    max_side = max(img.size)
    scale = 20 / max_side
    new_size = tuple([int(dim * scale) for dim in img.size])
    img = img.resize(new_size, Image.Resampling.LANCZOS)

    # ✅ 貼到黑底 28x28 上（置中）
    new_img = Image.new("L", (28, 28), 0)
    upper_left = ((28 - new_size[0]) // 2, (28 - new_size[1]) // 2)
    new_img.paste(img, upper_left)

    # ✅ 模糊 + 正規化
    img_array = np.array(new_img).astype(np.float32)
    img_array = gaussian_filter(img_array, sigma=0.4)
    img_array = np.clip(img_array, 0.0, 255.0)
    img_array /= 255.0

    # ✅ 顯示用
    plt.imshow(img_array, cmap='gray')
    plt.title("預處理後影像 (28x28)")
    plt.show()

    print("🖼️ 圖像總和:", np.sum(img_array))
    return img_array.reshape(1, 784)

def predict_digit():
    x = get_canvas_image()
    y = predict(x)
    result = np.argmax(y)

    result_label.config(text=f"辨識結果：{result}")

    top_k = y[0].argsort()[::-1][:3]
    probs = y[0][top_k]
    print("Top-3 預測：")
    for i in range(3):
        print(f"{i+1}. 數字 {top_k[i]}：{probs[i]:.2%}")

def clear_canvas():
    canvas.delete("all")
    result_label.config(text="辨識結果：")

def preview_image():
    get_canvas_image()

window = tk.Tk()
window.title("手寫數字辨識系統")
canvas = tk.Canvas(window, width=200, height=200, bg='white')
canvas.pack()
draw = ImageDraw.Draw(Image.new("RGB", (200, 200), (255, 255, 255)))

canvas.old_coords = None

def draw_lines(event):
    if canvas.old_coords:
        x1, y1 = canvas.old_coords
        x2, y2 = event.x, event.y
        canvas.create_line(x1, y1, x2, y2, width=10, fill='black', capstyle=tk.ROUND, smooth=True)
    canvas.old_coords = event.x, event.y

def reset_coords(event):
    canvas.old_coords = None

canvas.bind('<B1-Motion>', draw_lines)
canvas.bind('<ButtonRelease-1>', reset_coords)

btn_predict = tk.Button(window, text="辨識", command=predict_digit)
btn_predict.pack(side=tk.LEFT)

btn_clear = tk.Button(window, text="清除", command=clear_canvas)
btn_clear.pack(side=tk.LEFT)

btn_preview = tk.Button(window, text="預覽影像", command=preview_image)
btn_preview.pack(side=tk.LEFT)

result_label = tk.Label(window, text="辨識結果：", font=("Arial", 14))
result_label.pack()

window.mainloop()
