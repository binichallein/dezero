from turtle import forward
import numpy as np
from dezero.core import Function

def step_function(x):
    y = x > 0
    return y.astype(np.int32)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0,x)

class Sin(Function):

    def forward(self,x):
        y=np.sin(x)
        return y

    def backward(self,gy):
        x=self.inputs
        gx=gy*cos(x)
        return gx
    
class Cos(Function):
    def forward(self,x):
        y=np.cos(x)
        return y
    
    def backward(self,gy):
        x=self.inputs
        gx=gy*-sin(x)
        return gx

class Tanh(Function):
    def forward(self,x):
        y=np.tanh(x)
        return y
    
    def backward(self,gy):
        y=self.outputs[0]()
        gx=gy*(1-y**2)
        return gx

def tanh(x):
    return Tanh()(x)

def sin(x):
    return Sin()(x)

def cos(x):
    return Cos()(x)