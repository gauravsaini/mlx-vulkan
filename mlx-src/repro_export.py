import mlx.core as mx

mx.set_default_device(mx.gpu)
mx.random.seed(5)

def fun():
    return mx.random.uniform(shape=(3,))

print("Exporting...")
mx.export_function("fn.mlxfn", fun)
print("Importing...")
imported = mx.import_function("fn.mlxfn")
print("Running imported...")
out = imported()[0]
print("Done:", out.tolist())
