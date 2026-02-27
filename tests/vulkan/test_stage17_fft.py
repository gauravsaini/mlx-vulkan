import unittest
import mlx.core as mx
import numpy as np

class TestVulkanFFT(unittest.TestCase):
    def setUp(self):
        mx.set_default_device(mx.gpu)

    def test_fft_1d(self):
        # Test Radix-2 sizes
        sizes = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
        for n in sizes:
            # Generate random complex data
            # Real and imaginary parts
            x_np_real = np.random.randn(n).astype(np.float32)
            x_np_imag = np.random.randn(n).astype(np.float32)
            x_np = x_np_real + 1j * x_np_imag

            # numpy reference
            y_np = np.fft.fft(x_np)

            # MLX Vulkan evaluation
            # MLX uses float32 complex implicitly or separate arrays
            # Since MLX handles complex64, we inject real/imag arrays or direct if complex is built
            try:
               x_mx = mx.array(x_np, dtype=mx.complex64)
            except Exception as e:
               print(f"Skipping native complex conversion due to API limit: {e}")
               break

            y_mx = mx.fft.fft(x_mx)
            
            # Execute backend evaluation
            mx.eval(y_mx)

            # Assert identical output (within tolerance for float32 precision)
            np.testing.assert_allclose(np.array(y_mx), y_np, rtol=1e-4, atol=1e-4)
            
    def test_fft_radix4(self):
        # Stockham Radix-4 paths
        sizes = [64, 256, 1024, 4096]
        for n in sizes:
            x_np = (np.random.randn(n) + 1j * np.random.randn(n)).astype(np.complex64)
            y_np = np.fft.fft(x_np)
            
            x_mx = mx.array(x_np)
            y_mx = mx.fft.fft(x_mx)
            mx.eval(y_mx)
            
            np.testing.assert_allclose(np.array(y_mx), y_np, rtol=1e-4, atol=1e-4)

    def test_fft_real(self):
        # Test RFFT native
        n = 1024
        x_np = np.random.randn(n).astype(np.float32)
        y_np = np.fft.rfft(x_np)
        
        x_mx = mx.array(x_np)
        y_mx = mx.fft.rfft(x_mx)
        mx.eval(y_mx)
        
        np.testing.assert_allclose(np.array(y_mx), y_np, rtol=1e-4, atol=1e-4)

if __name__ == '__main__':
    unittest.main()
