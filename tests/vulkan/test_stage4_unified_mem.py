"""
Stage 4: Unified Memory / Zero-Copy Detection
Tests whether the GPU supports VK_EXT_external_memory_host (zero-copy for APUs)
and VK_KHR_external_memory for host-visible device-local memory.

This is critical for AMD APUs (like Strix Halo) where CPU+GPU share RAM.
With zero-copy, MLX array data never moves - GPU gets direct pointer to CPU alloc.

Run: python tests/vulkan/test_stage4_unified_mem.py
Pass: Reports whether zero-copy is available (not required to pass for basic ops)
"""
import sys
import subprocess
import json

def run():
    print("═══════════════════════════════════════")
    print("  STAGE 4: Unified Memory / Zero-Copy Detection")
    print("═══════════════════════════════════════")

    # Parse vulkaninfo JSON to check extensions
    try:
        result = subprocess.run(
            ["vulkaninfo", "--json"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            print("⚠️  vulkaninfo --json failed, using summary mode")
            return check_via_summary()
    except Exception as e:
        print(f"⚠️  vulkaninfo failed: {e}")
        return True  # Non-fatal

    try:
        data = json.loads(result.stdout)
        devices = data.get("devices", []) or data.get("Devices", [])

        for dev_info in devices:
            props = dev_info.get("properties", {})
            dev_name = props.get("deviceName", "Unknown")
            dev_type = props.get("deviceType", "")

            print(f"\nDevice: {dev_name}")
            print(f"Type:   {dev_type}")

            # Check for unified memory (integrated or virtual GPU)
            is_unified = dev_type in ["PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU",
                                       "VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU",
                                       "1"]  # Integrated = 1

            # Check extensions
            extensions = dev_info.get("extensions", [])
            ext_names = [e.get("extensionName", e) if isinstance(e, dict) else str(e)
                        for e in extensions]

            has_ext_memory_host = any("external_memory_host" in e for e in ext_names)
            has_ext_memory      = any("external_memory" in e and "host" not in e for e in ext_names)
            has_timeline_sem    = any("timeline_semaphore" in e for e in ext_names)

            print(f"\n  Extension Support:")
            icon = lambda b: "✅" if b else "❌"
            print(f"  {icon(is_unified)} Unified/Integrated Memory (zero-copy candidate)")
            print(f"  {icon(has_ext_memory_host)} VK_EXT_external_memory_host (Zero-Copy)")
            print(f"  {icon(has_ext_memory)} VK_KHR_external_memory")
            print(f"  {icon(has_timeline_sem)} VK_KHR_timeline_semaphore")

            if has_ext_memory_host and is_unified:
                print("\n  🚀 ZERO-COPY AVAILABLE!")
                print("  → device.h should use VK_EXT_external_memory_host for all allocations")
                print("  → allocator.cpp: import_host_memory() instead of vmaCreateBuffer()")
                print("  → Expected speedup: eliminate ALL CPU->GPU data transfers")
            elif has_ext_memory_host:
                print("\n  ⚡ VK_EXT_external_memory_host available (discrete GPU)")
                print("  → Can use for staging buffer optimization")
            else:
                print("\n  ℹ️  Standard memory model (copy required for CPU<->GPU)")

    except (json.JSONDecodeError, KeyError) as e:
        print(f"  ⚠️  Could not parse vulkaninfo JSON: {e}")
        return check_via_summary()

    print("\n✅ STAGE 4 COMPLETE (informational - non-blocking)")
    return True

def check_via_summary():
    """Fallback: check via vulkaninfo text output"""
    result = subprocess.run(
        ["vulkaninfo", "--summary"],
        capture_output=True, text=True, timeout=10
    )
    output = result.stdout + result.stderr
    print("\nVulkan Summary:")
    for line in output.split("\n"):
        if any(k in line for k in ["GPU", "deviceName", "deviceType", "apiVersion"]):
            print(f"  {line.strip()}")
    return True

if __name__ == "__main__":
    success = run()
    sys.exit(0 if success else 1)
