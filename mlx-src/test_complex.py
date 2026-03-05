import mlx.core as mx
import numpy as np

mx.set_default_device(mx.gpu)

for ax in [None, 0, 1, 2]:
    np.random.seed(42)
    a_np = np.arange(16).reshape(2, 2, 4).astype(np.int32)
    a_mlx = mx.array(a_np)

    if ax is None:
        idx_np = np.random.permutation(a_np.size)
        values_np = np.random.randint(low=0, high=100, size=(16,))
    else:
        shape = list(a_np.shape)
        shape[ax] = 2
        idx_np = np.random.choice(a_np.shape[ax], replace=False, size=(2,))
        idx_np = np.expand_dims(idx_np, list(range(1, 2 - ax + 1)))
        idx_np = np.broadcast_to(idx_np, shape)
        values_np = np.random.randint(low=0, high=100, size=shape)

    idx_mlx = mx.array(idx_np)
    values_mlx = mx.array(values_np)

    np.put_along_axis(a_np, idx_np, values_np, axis=ax)
    out_mlx = mx.put_along_axis(a_mlx, idx_mlx, values_mlx, axis=ax)
    match = np.array_equal(a_np, out_mlx)
    print(f"axis={ax}: {'PASS' if match else 'FAIL'}")
    if not match:
        print("  np:", a_np.flatten().tolist())
        print("  mx:", np.array(out_mlx).flatten().tolist())
