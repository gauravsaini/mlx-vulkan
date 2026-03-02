import sys; sys.path.insert(0,'python')
import mlx.core as mx, numpy as np
c = mx.matmul(mx.ones((4,4)), mx.ones((4,4)), stream=mx.gpu)
mx.eval(c)
print('first:', float(c[0,0].item()))
c2 = mx.matmul(mx.ones((4,4)), mx.ones((4,4)), stream=mx.gpu)
mx.eval(c2)
print('second:', float(c2[0,0].item()))
