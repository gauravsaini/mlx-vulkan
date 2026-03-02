#!/bin/bash
cd /Users/ektasaini/Desktop/mlx-vulkan/mlx-src
PYTHONPATH=python python3.11 test_ones.py > test_ones.log 2>&1 &
PID=$!
sleep 2
lldb -p $PID --batch -o "bt all" -o "quit" > lldb_bt.log 2>&1
kill -9 $PID
cat test_ones.log
echo "--- LLDB BT ---"
cat lldb_bt.log
