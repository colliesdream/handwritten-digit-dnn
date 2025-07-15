import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# 資料預處理
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 784) / 255.0
x_test = x_test.reshape(-1, 784) / 255.0
y_train_oh = np.eye(10)[y_train]
y_test_oh = np.eye(10)[y_test]

# He 初始化
def he_init(shape):
    return np.random.randn(*shape) * np.sqrt(2.0 / shape[0])

W1, b1 = he_init((784, 128)), np.zeros(128)
W2, b2 = he_init((128, 64)), np.zeros(64)
W3, b3 = he_init((64, 10)), np.zeros(10)

def relu(x): return np.maximum(0, x)
def relu_deriv(x): return (x > 0).astype(float)
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)
def cross_entropy(pred, true): return -np.sum(true * np.log(pred + 1e-8)) / len(true)

def forward(x):
    z1 = x @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    a2 = relu(z2)
    z3 = a2 @ W3 + b3
    a3 = softmax(z3)
    return z1, a1, z2, a2, z3, a3

# 超參數
lr = 0.01
batch_size = 100
epochs = 100
losses = []

# 訓練
for ep in range(epochs):
    indices = np.random.permutation(len(x_train))
    x_train, y_train_oh = x_train[indices], y_train_oh[indices]
    
    for i in range(0, len(x_train), batch_size):
        xb = x_train[i:i+batch_size]
        yb = y_train_oh[i:i+batch_size]

        z1, a1, z2, a2, z3, pred = forward(xb)

        dz3 = pred - yb
        dW3 = a2.T @ dz3 / batch_size
        db3 = dz3.mean(axis=0)

        dz2 = (dz3 @ W3.T) * relu_deriv(z2)
        dW2 = a1.T @ dz2 / batch_size
        db2 = dz2.mean(axis=0)

        dz1 = (dz2 @ W2.T) * relu_deriv(z1)
        dW1 = xb.T @ dz1 / batch_size
        db1 = dz1.mean(axis=0)

        W3 -= lr * dW3; b3 -= lr * db3
        W2 -= lr * dW2; b2 -= lr * db2
        W1 -= lr * dW1; b1 -= lr * db1

    _, _, _, _, _, train_pred = forward(x_train)
    loss = cross_entropy(train_pred, y_train_oh)
    losses.append(loss)
    print(f"Epoch {ep+1}/{epochs} - Loss: {loss:.4f}")

# 📉 Loss 曲線
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# ✅ 儲存模型
np.savez("dnn_model.npz", W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)
print("✅ 模型儲存完成為 dnn_model.npz")
