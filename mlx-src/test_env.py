import os
os.environ["MVK_CONFIG_SYNCHRONOUS_COMPILATION"] = "1"
os.environ["MVK_CONFIG_LOG_LEVEL"] = "3"
import sys; sys.path.insert(0,'python')
import mlx.core as mx

print("testing matmul...")
c = mx.matmul(mx.ones((4,4)), mx.ones((4,4)), stream=mx.gpu)
mx.eval(c)
print("first:", c[0,0].item())

c2 = mx.matmul(mx.ones((4,4)), mx.ones((4,4)), stream=mx.gpu)
mx.eval(c2)
print("second:", c2[0,0].item())
