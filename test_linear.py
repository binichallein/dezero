import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dezero import Variable
import dezero.functions as F

# 测试 Linear 层
print("=== Testing Linear Layer ===")

# 创建输入数据
x = Variable(np.random.rand(3, 4))  # batch_size=3, input_features=4
W = Variable(np.random.rand(4, 2))  # input_features=4, output_features=2
b = Variable(np.random.rand(2))     # output_features=2

print(f"x.shape: {x.shape}")
print(f"W.shape: {W.shape}")
print(f"b.shape: {b.shape}")

# 前向传播
y = F.linear(x, W, b)
print(f"y.shape: {y.shape}")
print(f"y = {y}")

# 测试反向传播
x.cleargrad()
W.cleargrad()
b.cleargrad()

loss = F.sum(y**2)
loss.backward()

print(f"\nGradients:")
print(f"x.grad.shape: {x.grad.shape}")
print(f"W.grad.shape: {W.grad.shape}")
print(f"b.grad.shape: {b.grad.shape}")

# 测试不带偏置的 Linear 层
print("\n=== Testing Linear Layer without bias ===")
x2 = Variable(np.random.rand(2, 3))
W2 = Variable(np.random.rand(3, 5))

y2 = F.linear(x2, W2)  # 不传 b 参数
print(f"y2.shape: {y2.shape}")

x2.cleargrad()
W2.cleargrad()

loss2 = F.sum(y2**2)
loss2.backward()

print(f"\nGradients (no bias):")
print(f"x2.grad.shape: {x2.grad.shape}")
print(f"W2.grad.shape: {W2.grad.shape}")

print("\n=== Testing with st2.py example ===")
# 使用 Linear 层重写 st2.py 的例子
np.random.seed(0)
x_data = np.random.rand(100, 1)
y_data = 5 + 2*x_data + np.random.rand(100, 1)

x = Variable(x_data)
y = Variable(y_data)

# 使用 Linear 层替代手动的矩阵乘法和加法
W = Variable(np.zeros((1, 1)))
b = Variable(np.zeros((1, 1)))

def predict_linear(x):
    return F.linear(x, W, b)

lr = 0.1
for i in range(10):
    W.cleargrad()
    b.cleargrad()

    y_pred = predict_linear(x)
    loss = F.mean_squared_error(y, y_pred)
    loss.backward()

    W.data -= lr * W.grad.data
    b.data -= lr * b.grad.data

    if i % 2 == 0:
        print(f"Iter {i}: Loss = {loss.data:.6f}, W = {W.data[0][0]:.6f}, b = {b.data[0][0]:.6f}")

print(f"\nFinal results with Linear layer:")
print(f"W = {W.data[0][0]:.6f} (target: ~2.0)")
print(f"b = {b.data[0][0]:.6f} (target: ~5.0)")