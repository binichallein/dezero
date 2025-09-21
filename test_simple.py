import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dezero import Variable
import dezero.functions as F

np.random.seed(0)
x=np.random.rand(100,1)
y=5+2*x+np.random.rand(100,1)
x=Variable(x)
y=Variable(y)

W=Variable(np.zeros((1,1)))
print(W)
b=Variable(np.zeros(1))

print(f"W type: {type(W)}, b type: {type(b)}")

def predict(x):
    y=F.matmul(x,W)+b
    return y

def mean_squared_error(x0,x1):
    diff =x0-x1
    return F.sum(diff**2)

lr=0.1
iters = 100

for i in range(iters):
    W.cleargrad()
    b.cleargrad()

    y_pred = predict(x)
    loss = mean_squared_error(y,y_pred)
    loss.backward()

    W.data -= lr*W.grad.data
    b.data -= lr*b.grad.data

    if i % 10 == 0:
        print(f"Iter {i}: Loss = {loss.data:.6f}")

print("Final results:")
print(f"W = {W.data}")
print(f"b = {b.data}")