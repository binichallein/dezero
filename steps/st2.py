if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__),'..'))

from tabnanny import verbose
import numpy as np
from dezero import Variable
from dezero.utils import plot_dot_graph
import math
import dezero.functions as F

np.random.seed(0)
x=np.random.rand(100,1)
y=5+2*x+np.random.rand(100,1)
x=Variable(x)
y=Variable(y)

W=Variable(np.zeros((1,1)))
print(W)
b=Variable(np.zeros((1,1)))

print(f"W type: {type(W)}, b type: {type(b)}")

def predict(x):
    
    return F.linear(x,W,b)


lr=0.1
iters = 200

for i in range(iters):
    W.cleargrad()
    b.cleargrad()

    y_pred = predict(x)
    loss = F.mean_squared_error(y,y_pred)
    loss.backward()

    W.data -= lr*W.grad.data
    b.data -= lr*b.grad.data

    if i % 10 == 0:
        print(f"Iter {i}: Loss = {loss.data:.6f}, W = {W.data[0][0]:.6f}, b = {b.data[0][0]:.6f}")

print("Final results:")
print(f"W = {W.data[0][0]:.6f} (target: ~2.0)")
print(f"b = {b.data[0][0]:.6f} (target: ~5.0)")