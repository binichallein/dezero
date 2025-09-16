from pyclbr import Function
import numpy as np
class Variable:
    def __init__(self, data):
        self.data = data
class Function:
    def __call__(self,input):
        data= input.data
        y= data**2
        output = Variable(y)
        return output
        
data = np.array(1.0)
x = Variable(data)

f= Function()
y=f(x)
print(type(y))
print(y.data)