import mlx.core as mx
import numpy as np

# Test BF16 add
a = mx.array([1.0, 2.0, 3.0], dtype=mx.bfloat16)
b = mx.array([4.0, 5.0, 6.0], dtype=mx.bfloat16)
c = a + b
mx.eval(c)
result = np.array(c.astype(mx.float32), dtype=np.float32)
print('BF16 add:', result)
assert np.allclose(result, [5.0, 7.0, 9.0], atol=0.1), f'BF16 add failed: {result}'
print('BF16 add: PASS')

# Test BF16 unary
a = mx.array([0.0, 1.0, 2.0], dtype=mx.bfloat16)
b = mx.exp(a)
mx.eval(b)
result = np.array(b.astype(mx.float32), dtype=np.float32)
expected = np.exp([0.0, 1.0, 2.0])
print('BF16 exp:', result, 'expected:', expected)
assert np.allclose(result, expected, atol=0.05), f'BF16 exp failed: {result}'
print('BF16 exp: PASS')
