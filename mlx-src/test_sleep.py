import mlx.core as mx
import time

try:
    print("Testing Sum 1")
    s1 = mx.sum(mx.array([1.0, 1.0]))
    mx.eval(s1)
    print(f"s1: {s1}")

    time.sleep(2)

    print("Testing Sum 2")
    s2 = mx.sum(mx.array([2.0, 2.0]))
    mx.eval(s2)
    print(f"s2: {s2}")

except Exception as e:
    print(f"Error: {e}")
