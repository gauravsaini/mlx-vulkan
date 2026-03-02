import sys
import unittest
import traceback

sys.path.insert(0, "/Users/ektasaini/Desktop/mlx-vulkan/mlx-src/python")
try:
    from tests.test_ops import TestOps
except Exception as e:
    print(f"Failed to import TestOps: {e}")
    sys.exit(1)

# Get all test methods
test_methods = [method for method in dir(TestOps) if method.startswith('test_')]

passed = 0
failed = []

for method_name in test_methods:
    print(f"Running {method_name}...")
    suite = unittest.TestSuite()
    suite.addTest(TestOps(method_name))
    runner = unittest.TextTestRunner(stream=open('/dev/null', 'w'), verbosity=0)
    try:
        result = runner.run(suite)
        if result.wasSuccessful():
            passed += 1
            print(f"  ✅ PASS")
        else:
            failed.append(method_name)
            print(f"  ❌ FAIL")
    except Exception as e:
        failed.append(method_name)
        print(f"  💥 CRASH: {e}")

print(f"\nResults: {passed} passed, {len(failed)} failed")
with open("failed_individual.txt", "w") as f:
    for name in failed:
        f.write(f"{name}\n")
