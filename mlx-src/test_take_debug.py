import mlx.core as mx

l = [
    [[1, 3], [-2, -2], [-3, -2]],
    [[2, 4], [-3, 2], [-4, -2]],
    [[2, 3], [2, 4], [2, 1]],
    [[1, -5], [3, -1], [2, 3]],
]
a = mx.array(l)
indices = mx.array([0, 0, -2])
mx.set_default_device(mx.gpu)
axis_take = mx.take(a, indices, axis=1)
print(axis_take.tolist())
