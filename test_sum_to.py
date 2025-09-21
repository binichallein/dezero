import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dezero import Variable
import dezero.functions as F

# 测试具体的形状转换
x = Variable(np.ones((100, 1)))  # 模拟梯度形状
target_shape = (1,)  # b 的形状

print(f"x.shape = {x.shape}")
print(f"target_shape = {target_shape}")

result = F.sum_to(x, target_shape)
print(f"result = {result}, result.shape = {result.shape}")

# 测试反向传播
result.backward()
print(f"x.grad = {x.grad}")
print(f"x.grad.shape = {x.grad.shape}")