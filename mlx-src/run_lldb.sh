#!/bin/bash
MLX_DISABLE_METAL=1 PYTHONPATH=python lldb --batch -o "run" -o "bt" -o "quit" -- /opt/homebrew/opt/python@3.11/bin/python3.11 -m pytest python/tests/test_ops.py -k "test_scans"
