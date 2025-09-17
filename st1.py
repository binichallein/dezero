import numpy as np

class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data,np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))
        self.data = data
        self.grad = None
        self.creator = None
    
    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        funcs = [self.creator]
        while funcs:
            f= funcs.pop()
            x,y = f.input,f.output
            x.grad = f.backward(y.grad)
            if x.creator is not None:
                funcs.append(x.creator)

class Function:
    def __call__(self,input):
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y))
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
    x0 = Variable(as_array(x.data - eps))
    x1 = Variable(as_array(x.data + eps))
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2*eps)

def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)

def f(x):
    return square(exp(square(x)))

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

if __name__ == '__main__':
    # 示例用法
    data = np.array(0.5)
    x = Variable(data)
    y = f(x)
    y.backward()
    print(f"x.grad = {x.grad}")
    
    # 数值微分验证
    numerical_grad = numerical_diff(f, Variable(np.array(0.5)))
    print(f"数值微分结果: {numerical_grad}")
    print(f"解析微分结果: {x.grad}")
    print(f"差异: {abs(numerical_grad - x.grad)}")