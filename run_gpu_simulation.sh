#!/bin/bash
# GPU-accelerated TIC Parameter Recovery Runner
# This script sets up the environment and runs the GPU simulation

# Set up CUDA library paths
NVIDIA_LIB_PATHS=$(find "$(dirname "$0")/.venv/lib/python3.13/site-packages/nvidia" -type d -name "lib" 2>/dev/null | tr '\n' ':' | sed 's/:$//')
export LD_LIBRARY_PATH="$NVIDIA_LIB_PATHS:$LD_LIBRARY_PATH"

# Run the simulation with all passed arguments
"$(dirname "$0")/.venv/bin/python" "$(dirname "$0")/simulations/parameter_recovery_v4_gpu_fixed.py" "$@"
