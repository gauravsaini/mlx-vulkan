import mlx.core as mx

# Just checking if there is a way to see raw memory
a = mx.array([-1.0, 1.0, 0.0])
res = mx.logical_not(a)
# Let's inspect the repr string to see if the array size is weird
print("Logical Not repr:", repr(res))
print("Logical Not type:", type(res))
print("Logical Not itemsize:", res.itemsize)
