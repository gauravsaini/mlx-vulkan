import mlx.core as mx

# Force use of Vulkan
mx.set_default_device(mx.gpu)

@mx.compile
def f(x, y):
    return x + y

def main():
    print(f"Device: {mx.default_device()}")
    a = mx.array([1., 2., 3.], dtype=mx.float32)
    b = mx.array([4., 5., 6.], dtype=mx.float32)
    
    print("Inputs:")
    print("a =", a)
    print("b =", b)
    
    print("\nRunning compiled f(a, b)...")
    c = f(a, b)
    
    # Evaluate so the graph executes
    mx.eval(c)
    print("Output:", c)

if __name__ == "__main__":
    main()
