import numpy as np
import weakref
import contextlib

import dezero


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

    @property
    def T(self):
        return dezero.functions.transpose(self)

    def transpose(self):
        return dezero.functions.transpose(self)
    
    def reshape(self,*shape):
        if len(shape)==1 and isinstance(shape[0], (tuple,list)):
            shape = shape[0]
        return dezero.functions.reshape(self,shape)

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

    def backward(self, retain_grads=False,create_graph=False):
        if self.grad is None:
            self.grad = Variable(np.ones_like(self.data))
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

            with using_config('enable_backprop',create_graph):
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

class Mul(Function):
    def forward(self,x0,x1):
        y=x0*x1
        return y
    
    def backward(self,gy):
        x0,x1=self.inputs
        return gy*x1, gy*x0

def numerical_diff(f,x,eps=1e-4):
    x0 = Variable(as_array(x.data - eps))
    x1 = Variable(as_array(x.data + eps))
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2*eps)


def mul(x0,x1):
    x1 = as_array(x1)
    return Mul()(x0,x1)

def add(x0,x1):
    x1 = as_array(x1)
    return Add()(x0,x1)

def neg(x):
    return Neg()(x)

def sub(x0,x1):
    x1 = as_array(x1)
    return Sub()(x0,x1)

def rsub(x0,x1):
    x1 = as_array(x1)
    return Sub()(x1,x0)

def div(x0,x1):
    x1 = as_array(x1)
    return Div()(x0,x1)

def rdiv(x0,x1):
    x1=as_array(x1)
    return Div()(x1,x0)

def pow(x,c):
    return Pow(c)(x)

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

def as_variable(obj):
    if isinstance(obj,Variable):
        return obj
    return Variable(obj)

class Add(Function):
    def forward(self,x0,x1):
        y=x0+x1
        return y

    def backward(self,gy):
        return gy,gy

class Neg(Function):
    def forward(self,x):
        return -x

    def backward(self,gy):
        return -gy


class Sub(Function):
    def forward(self,x0,x1):
        y=x0-x1
        return y

    def backward(self,gy):
        return gy,-gy

class Div(Function):
    def forward(self,x0,x1):
        y=x0/x1
        return y

    def backward(self,gy):
        x0,x1=self.inputs
        return gy/x1,-gy*x0/(x1**2)

class Pow(Function):
    def __init__(self,c) -> None:
        self.c =c

    def forward(self,x):
        return x**self.c

    def backward(self, gy):
        x, = self.inputs
        return self.c*x**(self.c-1)*gy

def setup_variable():
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__add__=add
    Variable.__radd__ = add
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__rtruediv__ = rdiv
    Variable.__truediv__ = div
    Variable.__pow__ = pow