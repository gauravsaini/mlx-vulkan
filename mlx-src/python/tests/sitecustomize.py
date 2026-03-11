import importlib.util
import os
import sys


def _install_mlx_core_override():
    core_so = os.getenv("MLX_CORE_SO")
    if not core_so or "mlx.core" in sys.modules:
        return

    spec = importlib.util.spec_from_file_location("mlx.core", core_so)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load mlx.core override from {core_so}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules["mlx.core"] = module

    try:
        import mlx as mlx_pkg

        mlx_pkg.core = module
    except Exception:
        pass


_install_mlx_core_override()
