import subprocess
print("Running lldb...")
subprocess.run(["lldb", "-c", "core"], shell=False)
