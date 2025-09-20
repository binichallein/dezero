if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__),'..'))

from tabnanny import verbose
import numpy as np
from dezero import Variable
from dezero.utils import plot_dot_graph
import math
import dezero.functions as F

def sphere(x,y):
    return x**2+y**2

def matyas(x,y):
    return 0.26*(x**2+y**2)-0.48*x*y

def goldstein(x,y):
    return (1+(x+y+1)**2*(19-14*x+3*x**2-14*y+6*x*y+3*y**2))*(30+(2*x-3*y)**2*(18-32*x+12*x**2+48*y-36*x*y+27*y**2))

def my_sin(x,thershold=0.0001):
    y=0
    for i in range(100000):
        c=(-1)**i/math.factorial(2*i+1)
        t=c*x**(2*i+1)
        y+=t
        if abs(t.data)<thershold:
            break
    return y

def rosenbrock(x0,x1):
    y=100*(x1-x0**2)**2+(x0-1)**2
    return y

def f(x):
    y= x**4-2*x**2
    return y

def gx2(x):
    return 12*x**2-4


def te(x):
    return x**2

# y=te(x)
# y.backward(create_graph=True)
# print(x.grad)
# gx = x.grad
# x.cleargrad()
# gx.backward()
# print(x.grad)

# print(gx)

# 牛顿法
# iters=10

# for i in range(iters):
#     print(i,x)
#     x.cleargrad()
#     y=f(x)
#     y.backward(create_graph=True)
#     gx = x.grad
#     x.cleargrad()
#     gx.backward()
#     gx2 = x.grad
#     x.data -= gx.data/gx2.data
    
# # z=sphere(x,y)

# # z=matyas(x,y)
# z=goldstein(x,y)
# z.name='z'
# z.backward()
# plot_dot_graph(z,verbose=False,to_file='goldstein.png')

# lr=0.00005
# i=0
# while True:
#     i+=1
#     x0.cleargrad()
#     x1.cleargrad()
#     y= rosenbrock(x0,x1)
#     y.backward()
#     x0.data -= lr*x0.grad
#     x1.data -= lr*x1.grad
#     # 检查梯度是否足够小（数值精度问题）
#     if abs(x0.grad) < 1e-10 and abs(x1.grad) < 1e-10:
#         print(f"收敛到: x0={x0.data}, x1={x1.data}, 梯度: x0.grad={x0.grad}, x1.grad={x1.grad}")
#         break
#     if i%10000==0:
#         print(f"第{i}次迭代: x0={x0.data}, x1={x1.data}, 梯度: x0.grad={x0.grad}, x1.grad={x1.grad}")
    

# y = F.tanh(x)
# x.name='x'
# y.name = 'y'
# y.backward(create_graph=True)

# iters = 5

# for i in range(iters):
#     gx = x.grad
#     x.cleargrad()
#     gx.backward(create_graph=True)
  

# gx = x.grad
# gx.name = 'gx' + str(iters+1)
# plot_dot_graph(gx,verbose=False,to_file='tanh6.png')

x = Variable(np.array([[1,2,3],[4,5,6]]))

a = Variable(np.random.randn(1,2,3))
# print(a)
# y=a.reshape(2,3)
# print(y)
# y=a.reshape((2,3))
# print(y)
# y=a.reshape([2,3])
# print(y)
print(x.T)
print(x.transpose())

# y=F.reshape(x,(6,))
# y.backward()

# print(x.grad)