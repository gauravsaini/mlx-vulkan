#!/usr/bin/env python3
"""
Regression test for Vulkan compiled-graph fallback policy.

Default behavior should warn once and run the compiled graph on CPU.
Strict mode should fail loudly when a compiled graph reaches the Vulkan backend.
"""

import os
import subprocess
import sys
import tempfile
import textwrap


def run_case(code, extra_env=None):
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as handle:
        handle.write(code)
        script_path = handle.name
    try:
        return subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            env=env,
        )
    finally:
        os.unlink(script_path)


def compiled_snippet():
    return textwrap.dedent(
        """
        import importlib.util
        import os
        import sys

        core_so = os.environ.get("MLX_CORE_SO")
        if core_so:
            import mlx
            spec = importlib.util.spec_from_file_location("mlx.core", core_so)
            mx = importlib.util.module_from_spec(spec)
            sys.modules["mlx.core"] = mx
            spec.loader.exec_module(mx)
            mlx.core = mx
        else:
            import mlx.core as mx

        if not mx.is_available(mx.gpu):
            raise RuntimeError("mx.gpu not available")

        fn = mx.compile(lambda x: (x + 1.0) * 2.0)
        with mx.stream(mx.gpu):
            x = mx.array([1.0, 2.0, 3.0], dtype=mx.float32)
            out = fn(x)
            mx.eval(out)
        print(out.tolist())
        """
    )


def test_warning_mode():
    proc = run_case(compiled_snippet())
    if proc.returncode != 0:
        raise AssertionError(
            f"default compiled fallback unexpectedly failed\nstdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    if "[4.0, 6.0, 8.0]" not in proc.stdout:
        raise AssertionError(f"unexpected compiled output:\n{proc.stdout}")
    expected = (
        "[MLX Vulkan] WARNING: mx.compile() graphs have no Vulkan implementation "
        "yet; falling back to CPU."
    )
    if expected not in proc.stderr:
        raise AssertionError(
            "missing compiled fallback warning\n"
            f"stderr was:\n{proc.stderr}"
        )


def test_strict_mode():
    proc = run_case(
        compiled_snippet(),
        {"MLX_VULKAN_FAIL_ON_CPU_FALLBACK": "1"},
    )
    if proc.returncode == 0:
        raise AssertionError(
            "strict compiled fallback unexpectedly succeeded\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    expected = (
        "Compiled Vulkan graphs are not supported while "
        "MLX_VULKAN_FAIL_ON_CPU_FALLBACK=1."
    )
    combined = proc.stdout + proc.stderr
    if expected not in combined:
        raise AssertionError(
            "missing strict compiled fallback error\n"
            f"output was:\n{combined}"
        )


def main():
    test_warning_mode()
    print("  PASS: default compiled fallback warns once")
    test_strict_mode()
    print("  PASS: strict compiled fallback errors loudly")
    print("PASS: Vulkan compiled fallback policy succeeded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
