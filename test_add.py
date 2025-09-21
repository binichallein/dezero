import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dezero import Variable
import dezero.functions as F

# 测试简单的加法
a = Variable(np.array([[1.0, 2.0]]))  # shape (1, 2)
b = Variable(np.array([[3.0]]))       # shape (1, 1)

print(f"a.shape = {a.shape}, b.shape = {b.shape}")

c = a + b
print(f"c = {c}, c.shape = {c.shape}")

c.backward()
print(f"a.grad = {a.grad}")
print(f"b.grad = {b.grad}")