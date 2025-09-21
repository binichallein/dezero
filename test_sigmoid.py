import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dezero import Variable
import dezero.functions as F

print("=== Testing Sigmoid Function ===")

# 测试 sigmoid 函数
x = Variable(np.array([0.0, 1.0, -1.0, 2.0, -2.0]))
print(f"x = {x.data}")

y = F.sigmoid(x)
print(f"sigmoid(x) = {y.data}")

# 验证 sigmoid 的数学性质
print(f"sigmoid(0) = {F.sigmoid(Variable(np.array([0.0]))).data[0]:.6f} (should be ~0.5)")
print(f"sigmoid(large positive) = {F.sigmoid(Variable(np.array([10.0]))).data[0]:.6f} (should be ~1.0)")
print(f"sigmoid(large negative) = {F.sigmoid(Variable(np.array([-10.0]))).data[0]:.6f} (should be ~0.0)")

# 测试反向传播
x.cleargrad()
loss = F.sum(y**2)
loss.backward()

print(f"\nGradients:")
print(f"x.grad = {x.grad.data}")

print("\n=== Testing Exp Function ===")

# 测试 exp 函数
x2 = Variable(np.array([0.0, 1.0, -1.0, 2.0]))
print(f"x2 = {x2.data}")

y2 = F.exp(x2)
print(f"exp(x2) = {y2.data}")

# 验证 exp 的数学性质
print(f"exp(0) = {F.exp(Variable(np.array([0.0]))).data[0]:.6f} (should be 1.0)")
print(f"exp(1) = {F.exp(Variable(np.array([1.0]))).data[0]:.6f} (should be ~2.718)")

# 测试 exp 的反向传播
x2.cleargrad()
loss2 = F.sum(y2)
loss2.backward()

print(f"\nGradients for exp:")
print(f"x2.grad = {x2.grad.data}")

print("\n=== Testing Sigmoid with Neural Network ===")

# 简单的二分类例子
np.random.seed(42)
# 生成线性可分的二分类数据
X = np.random.randn(100, 2)
y_true = (X[:, 0] + X[:, 1] > 0).astype(np.float64).reshape(-1, 1)

X = Variable(X)
y_true = Variable(y_true)

# 参数
W = Variable(np.random.randn(2, 1) * 0.1)
b = Variable(np.zeros((1, 1)))

def predict(x):
    logits = F.linear(x, W, b)
    probs = F.sigmoid(logits)
    return probs

def binary_cross_entropy(y_pred, y_true):
    # 简化版本的二元交叉熵
    eps = 1e-15  # 防止log(0)
    y_pred_clipped = y_pred * (1 - 2*eps) + eps
    return F.sum(-(y_true * F.log(y_pred_clipped) + (1 - y_true) * F.log(1 - y_pred_clipped)))

print("Training binary classifier with sigmoid...")
lr = 0.1
for epoch in range(50):
    W.cleargrad()
    b.cleargrad()

    y_pred = predict(X)

    # 使用均方误差代替交叉熵（因为还没实现log）
    loss = F.mean_squared_error(y_pred, y_true)
    loss.backward()

    W.data -= lr * W.grad.data
    b.data -= lr * b.grad.data

    if epoch % 10 == 0:
        accuracy = np.mean((y_pred.data > 0.5) == y_true.data)
        print(f"Epoch {epoch}: Loss = {loss.data:.6f}, Accuracy = {accuracy:.3f}")

print(f"\nFinal weights: W = {W.data.flatten()}")
print(f"Final bias: b = {b.data.flatten()}")

# 测试最终预测
test_points = np.array([[1.0, 1.0], [-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0]])
test_X = Variable(test_points)
test_pred = predict(test_X)
print(f"\nTest predictions:")
for i, (point, pred) in enumerate(zip(test_points, test_pred.data)):
    print(f"Point {point}: {pred[0]:.3f} ({'Positive' if pred[0] > 0.5 else 'Negative'})")