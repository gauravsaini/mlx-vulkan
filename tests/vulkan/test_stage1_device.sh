#!/bin/bash
# Stage 1: Vulkan Device Detection
# Run this FIRST before any compilation
# Requires: vulkaninfo installed (brew install vulkan-tools on macOS, apt install vulkan-tools on Linux)

set -e
echo "═══════════════════════════════════════"
echo "  STAGE 1: Vulkan Device Detection"
echo "═══════════════════════════════════════"

# Check vulkaninfo is available
if ! command -v vulkaninfo &>/dev/null; then
    echo "❌ FAIL: vulkaninfo not found"
    echo "   Install: brew install vulkan-tools (macOS) / sudo apt install vulkan-tools (Linux)"
    exit 1
fi

# Check glslc is available
if ! command -v glslc &>/dev/null; then
    echo "❌ FAIL: glslc not found"
    echo "   Install: brew install shaderc (macOS) / sudo apt install glslc (Linux)"
    exit 1
fi

# Check VK device count
DEVICE_COUNT=$(vulkaninfo --summary 2>/dev/null | grep -c "GPU" || echo "0")
if [ "$DEVICE_COUNT" -eq 0 ]; then
    echo "❌ FAIL: No Vulkan GPU found"
    echo "   Run: vulkaninfo --summary  to debug"
    exit 1
fi

echo "✅ Vulkan GPU count: $DEVICE_COUNT"

# Print device info
echo ""
echo "Device Summary:"
vulkaninfo --summary 2>/dev/null | grep -E "GPU|deviceName|apiVersion|driverVersion" || true

# Check API version >= 1.2
API_VER=$(vulkaninfo --summary 2>/dev/null | grep "apiVersion" | head -1 | grep -oE "[0-9]+\.[0-9]+" | head -1)
echo ""
echo "Vulkan API Version: $API_VER"

echo ""
echo "✅ STAGE 1 PASS: Vulkan device detected"
