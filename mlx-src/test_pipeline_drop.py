import mlx.core as mx

try:
    print("Testing Sum 1")
    s1 = mx.sum(mx.array([5.0, 5.0]))
    mx.eval(s1)
    print(f"s1 (first ever): {s1}")

    print("Testing Sum 2")
    s2 = mx.sum(mx.array([5.0, 5.0]))
    mx.eval(s2)
    print(f"s2 (second ever): {s2}")

    print("Testing Max 1")
    m1 = mx.max(mx.array([5.0, 5.0]))
    mx.eval(m1)
    print(f"m1 (first max): {m1}")

except Exception as e:
    print(f"Error: {e}")
