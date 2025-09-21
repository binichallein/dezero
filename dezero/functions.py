from turtle import forward
import numpy as np
from dezero import utils
from dezero.core import Function
from dezero.core import as_variable

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

class Reshape(Function):
    def __init__(self,shape):
        self.shape = shape

    def forward(self,x):
        self.x_shape = x.shape
        y=x.reshape(self.shape)
        return y

    def backward(self,gy):
        return reshape(gy,self.x_shape)

class Transpose(Function):

    def forward(self,x):
        y=np.transpose(x)
        return y
    
    def backward(self,gy):
        gx = transpose(gy)
        return gx

class Sum(Function):
    def __init__(self,axis=None,keepdims=False):
        self.axis = axis
        self.keepdims = keepdims


    def forward(self,x):
        self.x_shape = x.shape
        y=x.sum(axis=self.axis,keepdims=self.keepdims)
        return y

    def backward(self, gy):
        gy = utils.reshape_sum_backward(gy,self.x_shape,self.axis,self.keepdims)
        return broadcast_to(gy,self.x_shape)


class BroadcastTo(Function):
    def __init__(self,shape):
        self.shape = shape

    def forward(self,x):
        self.x_shape = x.shape
        y=np.broadcast_to(x,self.shape)
        return y

    def backward(self,gy):
        return sum_to(gy,self.x_shape)

def broadcast_to(x,shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)

class SumTo(Function):
    def __init__(self,shape):
        self.shape = shape

    def forward(self,x):
        self.x_shape = x.shape
        y=utils.sum_to(x,self.shape)
        return y

    def backward(self,gy):
        return broadcast_to(gy,self.x_shape)

class MatMul(Function):
    def forward(self,x,W):
        return x.dot(W)

    def backward(self, gy):
        x,W = self.inputs
        gx = matmul(gy,W.T)
        gW = matmul(x.T,gy)
        return gx,gW

class MeanSquaredError(Function):
    def forward(self,x0,x1):
        diff = x0-x1
        self.diff_shape = diff.shape
        return (diff**2).sum()/diff.shape[0]

    def backward(self,gy):
        x0,x1 = self.inputs
        diff = x0-x1
        n = self.diff_shape[0]
        gx0 = gy*diff*(2./n)
        gx1=-gx0
        return gx0,gx1

def mean_squared_error(x0,x1):
    return MeanSquaredError()(x0,x1)


def matmul(x,W):
    return MatMul()(x,W)

def sum_to(x,shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)

def sum(x,axis=None,keepdims=False):
    return Sum(axis,keepdims)(x)

def transpose(x):
    return Transpose()(x)

def reshape(x,shape):
    if x.shape == shape:
        return as_variable(x)
    else:
        return Reshape(shape)(x)

def tanh(x):
    return Tanh()(x)

def sin(x):
    return Sin()(x)

def cos(x):
    return Cos()(x)

class Linear(Function):
    def forward(self, x, W, b=None):
        y = x.dot(W)
        if b is not None:
            y += b
        return y

    def backward(self, gy):
        x, W = self.inputs[:2]

        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)

        if len(self.inputs) == 3:  # 有偏置
            b = self.inputs[2]
            # 对于偏置，需要沿着 batch 维度求和
            gb = sum(gy, axis=0)
            gb = sum_to(gb, b.shape)
            return gx, gW, gb
        else:  # 没有偏置
            return gx, gW

def linear(x, W, b=None):
    if b is None:
        return Linear()(x, W)
    else:
        return Linear()(x, W, b)