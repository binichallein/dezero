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
y=np.sin(2*np.pi*x)+np.random.rand(100,1)

I,H,O=1,10,1
W1=Variable(0.01*np.random.randn(I,H))
b1 = Variable(np.zeros(H))
W2=Variable(0.01*np.random.randn(H,O))
b2 = Variable(np.zeros(O))


def predict(x):
    
    y= F.linear(x,W1,b1)
    y=F.sigmoid(y)
    y=F.linear(y,W2,b2)
    return y


lr=0.2
iters = 100000

for i in range(iters):

    W1.cleargrad()
    W2.cleargrad()
    b1.cleargrad()
    b2.cleargrad()

    y_pred = predict(x)
    loss = F.mean_squared_error(y,y_pred)
    loss.backward()
    
    W1.data -= lr*W1.grad.data
    W2.data -= lr*W2.grad.data
    b1.data -= lr*b1.grad.data
    b2.data -= lr*b2.grad.data

    if i % 1000 == 0:
        print(f"Iter {i}: Loss = {loss.data:.6f}")

print("Final results:")
print(f"W1 = {W1.data}")
print(f"W2 = {W2.data}")
print(f"b1 = {b1.data}")
print(f"b2 = {b2.data}")
