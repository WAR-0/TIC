# GPU Setup for TIC Simulations

## Installation Complete ✓

The GPU-accelerated simulation environment has been successfully set up with:

- Python 3.13 virtual environment (`.venv`)
- JAX with CUDA 12 support
- Optax optimizer
- NumPy and SciPy

## GPU Detected

Your NVIDIA RTX GPU is detected and working with JAX!

## Running Simulations

### Quick Start

Use the provided run script:

```bash
./run_gpu_simulation.sh --n-sim 500 --n-participants 35 --max-steps 1500
```

### Manual Activation

If you prefer to activate the environment manually:

```bash
source .venv/bin/activate_with_cuda
python simulations/parameter_recovery_v4_gpu_fixed.py --n-sim 500 --n-participants 35
```

### Command-Line Options

- `--n-sim N`: Number of simulations (default: 200)
- `--n-participants N`: Participants per simulation (default: 35)
- `--max-steps N`: Optimization steps (default: 1500)
- `--learning-rate FLOAT`: Adam learning rate (default: 0.01)
- `--tol FLOAT`: Gradient norm tolerance (default: 1e-3)
- `--log-interval N`: Status print interval (default: 200 steps)
- `--seed N`: Random seed (default: 42)

## Results

Results are saved to: `simulations/results_v4_gpu.txt`

## Latest Run

**500 simulations completed successfully!**

Key statistics:
- D0: r = -0.569, RelBias = 50.9%
- lambda: r = 0.079, RelBias = 11.2%
- kappa: r = 0.356, RelBias = 10.7%
- gamma: r = 0.139, RelBias = -31.9%
- rho: r = 0.625, RelBias = 0.4%

## Technical Notes

### Fixed Version

The original `parameter_recovery_v4_gpu.py` had a JAX jitting issue with the `SimulationBatch` dataclass. The fixed version (`parameter_recovery_v4_gpu_fixed.py`) uses tuple unpacking instead, which is compatible with JAX's JIT compilation.

### CUDA Libraries

The setup uses pip-installed CUDA libraries bundled with the JAX CUDA plugin. Library paths are automatically configured by the `run_gpu_simulation.sh` script.

## Troubleshooting

If GPU detection fails, ensure:
1. NVIDIA drivers are installed: `nvidia-smi`
2. CUDA toolkit is available: `nvcc --version`
3. LD_LIBRARY_PATH includes CUDA libraries (handled by the run script)

