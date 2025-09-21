import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dezero import Variable
import dezero.functions as F

# 模拟真实场景
x = Variable(np.random.rand(100, 1))
W = Variable(np.zeros((1, 1)))
b = Variable(np.zeros(1))

print(f"x.shape = {x.shape}")
print(f"W.shape = {W.shape}")
print(f"b.shape = {b.shape}")

# 前向传播
matmul_result = F.matmul(x, W)
print(f"matmul_result.shape = {matmul_result.shape}")

result = matmul_result + b
print(f"result.shape = {result.shape}")

# 反向传播
# 不调用 cleargrad()
result.backward()

print(f"W.grad = {W.grad}")
print(f"b.grad = {b.grad}")