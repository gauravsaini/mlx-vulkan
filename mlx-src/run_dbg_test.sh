#!/bin/bash
export MLX_DISABLE_METAL=1
export PYTHONPATH=python
/opt/homebrew/Frameworks/Python.framework/Versions/3.14/bin/python3.14 -c "import mlx.core as mx; mx.ones(10)"
