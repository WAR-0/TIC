#!/bin/bash
# GPU-accelerated TIC Parameter Recovery Runner (Interactive Mode)
# Shows real-time progress and intermediate statistics

# Set up CUDA library paths
NVIDIA_LIB_PATHS=$(find "$(dirname "$0")/.venv/lib/python3.13/site-packages/nvidia" -type d -name "lib" 2>/dev/null | tr '\n' ':' | sed 's/:$//')
export LD_LIBRARY_PATH="$NVIDIA_LIB_PATHS:$LD_LIBRARY_PATH"

# Run the interactive simulation with all passed arguments
"$(dirname "$0")/.venv/bin/python" "$(dirname "$0")/simulations/parameter_recovery_v4_gpu_interactive.py" "$@"
