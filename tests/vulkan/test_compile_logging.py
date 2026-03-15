#!/usr/bin/env python3
"""
Regression test for Vulkan compiled-kernel logging.

When MLX_LOG_COMPILE_TAPE=1 is set, the Vulkan compiled backend should log a
unique kernel description once and print an aggregated summary at process exit.
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
            print("SKIP: mx.gpu not available")
            raise SystemExit(0)

        fn = mx.compile(lambda x: (x + 1.0) * 2.0)
        with mx.stream(mx.gpu):
            x = mx.array([1.0, 2.0, 3.0], dtype=mx.float32)
            y0 = fn(x)
            y1 = fn(x)
            mx.eval(y0, y1)
        print(y0.tolist(), y1.tolist())
        """
    )


def test_compile_logging():
    proc = run_case(compiled_snippet(), {"MLX_LOG_COMPILE_TAPE": "1"})
    if "SKIP: mx.gpu not available" in proc.stdout:
        return False
    if proc.returncode != 0:
        raise AssertionError(
            f"compiled logging run failed\nstdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )

    if "[2.0, 4.0, 6.0] [2.0, 4.0, 6.0]" not in proc.stdout:
        raise AssertionError(f"unexpected compiled output:\n{proc.stdout}")

    kernel_logs = [
        line
        for line in proc.stderr.splitlines()
        if line.startswith("[MLX Vulkan][compile profile] kernel=")
    ]
    if len(kernel_logs) != 1:
        raise AssertionError(
            "expected exactly one unique kernel log\n"
            f"stderr was:\n{proc.stderr}"
        )
    if "ops=Add -> Multiply" not in kernel_logs[0]:
        raise AssertionError(
            "kernel log missing fused op sequence\n"
            f"kernel log was:\n{kernel_logs[0]}"
        )

    summary_line = "[MLX Vulkan][compile profile] summary kernels=1 dispatches=2"
    if summary_line not in proc.stderr:
        raise AssertionError(
            "missing compile profile summary\n"
            f"stderr was:\n{proc.stderr}"
        )
    if "op=Add dispatches=2 unique_kernels=1" not in proc.stderr:
        raise AssertionError(
            "missing Add frequency summary\n"
            f"stderr was:\n{proc.stderr}"
        )
    if "op=Multiply dispatches=2 unique_kernels=1" not in proc.stderr:
        raise AssertionError(
            "missing Multiply frequency summary\n"
            f"stderr was:\n{proc.stderr}"
        )
    return True


def main():
    if not test_compile_logging():
        print("SKIP: Vulkan compile logging (mx.gpu not available)")
        return 0
    print("PASS: Vulkan compile logging succeeded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
