#!/bin/bash
PYTHONPATH=python python3.11 -c "import tests.test_ops as t; o=t.TestOps(); o.setUpClass(); o.test_binary()" > test_ops.log 2>&1 &
PID=$!
sleep 2
echo "Attaching to $PID"
lldb -p $PID --batch -o "continue" -o "bt" -o "quit" > lldb_bt.log 2>&1
kill -9 $PID || true
echo "--- LLDB BT ---"
cat lldb_bt.log
