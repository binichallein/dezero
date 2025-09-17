import unittest
import numpy as np
import sys
import os

# 添加父目录到路径，以便导入st1模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from st1 import Variable, square, exp

class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)

    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)

class ExpTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(1.0))
        y = exp(x)
        expected = np.array(np.e)
        np.testing.assert_allclose(y.data, expected)
    
    def test_backward(self):
        x = Variable(np.array(1.0))
        y = exp(x)
        y.backward()
        expected = np.array(np.e)
        np.testing.assert_allclose(x.grad, expected)

if __name__ == '__main__':
    unittest.main()