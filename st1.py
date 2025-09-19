import numpy as np
import weakref
import contextlib


class Variable:
    __array_priority__ = 200

    def __init__(self, data,name=None):
        if data is not None:
            if not isinstance(data,np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))
        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation =0
    

    def __repr__(self):
        if self.data is None:
            return "Variable(None)"
        p =str(self.data).replace('\n', '\n'+' '*9)
        return 'Variable('+p+', dtype='+str(self.dtype)+')'

    def __len__(self):
        return len(self.data)

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size
    
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def dtype(self):
        return self.data.dtype
    
    
    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1
    
    def cleargrad(self):
        self.grad = None

    def backward(self, retain_grads=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                seen_set.add(f)
                funcs.append(f)
                funcs.sort(key= lambda x: x.generation)
            
        add_func(self.creator)

        while funcs:
            f= funcs.pop()
            # x,y = f.input,f.output
            # x.grad = f.backward(y.grad)
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs,tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs,gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
                if x.creator is not None:
                    add_func(x.creator)

            if not retain_grads:
                for y in f.outputs:
                    y().grad=None


def as_variable(obj):
    if isinstance(obj,Variable):
        return obj
    return Variable(obj)

class Function:
    def __call__(self,*inputs):
        inputs = [as_variable(x) for x in inputs]
        xs = [x.data for x in inputs]
        ys = self.forward(*xs) 
        if not isinstance(ys,tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.outputs = [weakref.ref(output) for output in outputs]

            self.inputs = inputs

        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self,x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()

class Config:
    enable_backprop = True

@contextlib.contextmanager
def using_config(name,value):
    old_value=getattr(Config,name)
    setattr(Config,name,value)
    try:
        yield
    
    finally:
        setattr(Config,name,old_value)

def no_grad():
    return using_config('enable_backprop',False)

class Square(Function):
    def forward(self, x):
        return x**2
    
    def backward(self,gy):
        x= self.inputs[0].data
        return 2*x*gy

class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
    def backward(self,gy):
        x=self.inputs[0].data
        return np.exp(x)*gy

class Mul(Function):
    def forward(self,x0,x1):
        y=x0*x1
        return y
    
    def backward(self,gy):
        x0,x1=self.inputs[0].data, self.inputs[1].data
        return gy*x1, gy*x0

def numerical_diff(f,x,eps=1e-4):
    x0 = Variable(as_array(x.data - eps))
    x1 = Variable(as_array(x.data + eps))
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2*eps)

def square(x):
    return Square()(x)

def mul(x0,x1):
    x1 = as_array(x1)
    return Mul()(x0,x1)

def exp(x):
    return Exp()(x)

def add(x0,x1):
    x1 = as_array(x1)
    return Add()(x0,x1)

def f(x):
    return square(exp(square(x)))

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

class Add(Function):
    def forward(self,x0,x1):
        y=x0+x1
        return y

    def backward(self,gy):
        return gy,gy

Variable.__mul__ = mul
Variable.__rmul__ = mul
Variable.__add__=add
Variable.__radd__ = add

if __name__ == '__main__':
    # # 示例用法
    data = np.array(0.5)
    x = Variable(data)
    y = f(x)
    print(f"x.grad = {x.grad}")
    y.backward()
    print(f"x.grad = {x.grad}")
    
    # 数值微分验证
    numerical_grad = numerical_diff(f, Variable(np.array(0.5)))
    print(f"数值微分结果: {numerical_grad}")
    print(f"解析微分结果: {x.grad}")
    print(f"差异: {abs(numerical_grad - x.grad)}")
    x0 = Variable(np.array(1.0))
    x1 = Variable(np.array(1.0))

    t=add(x0,x1)
    y=add(x0,t)
    y.backward()
    print(("{},{}".format(x0.grad,x1.grad)))
    print(("{},{}".format(t.grad,y.grad)))


    a=Variable(np.array(2))
    # b=Variable(np.array(2))
    # c=Variable(np.array(1))
    # y=a*b+c
    # print(y)
    # y.backward(True)
    # print(a.grad)
    # print(b.grad)
    # print(c.grad)
    y = np.array(1.0)+a
    print(y)