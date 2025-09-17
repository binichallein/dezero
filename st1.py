import numpy as np
class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None
    
    def set_creator(self, func):
        self.creator = func

    def backward(self):
        f = self.creator
        if f is not None:
            f.backward(self.grad)
        else:
            raise RuntimeError("creator not found")

class Function:
    def __call__(self,input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self)
        self.output = output
        self.input = input
        return output
    
    def forward(self,x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()

class Square(Function):
    def forward(self, x):
        return x**2
    
    def backward(self,gy):
        x= self.input.data
        return 2*x*gy

class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
    def backward(self,gy):
        x=self.input.data
        return np.exp(x)*gy

def numerical_diff(f,x,eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2*eps)

def f(x):
    A=Square()
    B=Exp()
    C=Square()
    return C(B(A(x)))
A=Square()
B=Exp()
C=Square()
data = np.array(0.5)

x = Variable(data)
a=A(x)
b=B(a)
c=C(b)

c.grad = np.array(1.0)
b.grad = C.backward(c.grad)
a.grad = B.backward(b.grad)
x.grad = A.backward(a.grad)
print(x.grad)