import sys; sys.path.insert(0,'python')
import mlx.core as mx
print("testing ones...")
a = mx.ones((4,4), stream=mx.gpu)
mx.eval(a)
print("ones returned:", a[0,0].item())
