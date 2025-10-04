# Interactive GPU Simulation Mode

## Overview

The interactive mode provides **real-time progress monitoring** with intermediate statistics updates, allowing you to monitor parameter recovery quality as simulations run and stop early if needed.

## Quick Start

```bash
./run_gpu_simulation_interactive.sh --n-sim 100 --update-interval 10
```

## Features

✓ **Progress bars** for each optimization run  
✓ **Live statistics** updated every N simulations  
✓ **Correlation tracking** to monitor recovery quality  
✓ **Early stopping** capability (Ctrl+C if results look poor)  

## Example Output

```
Simulation 5/100
Optimizing |██████████████████████████████| 100.0% loss= 60082.8 grad= 45.50

==========================================================================================
INTERIM RESULTS (5/100 simulations)
==========================================================================================
Parameter           r       Bias   RelBias%       RMSE        MAE
------------------------------------------------------------------------------------------
D0            -0.849     0.090      60.0     0.093     0.090
lambda        -0.058     0.287      11.5     0.419     0.328
kappa          0.230     0.199      10.5     0.270     0.220
gamma          0.196    -0.575     -31.9     0.615     0.575
rho            0.522     0.004       0.1     0.148     0.111
==========================================================================================
```

## Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--n-sim` | 200 | Number of simulations to run |
| `--n-participants` | 35 | Participants per simulation |
| `--max-steps` | 1500 | Maximum optimization steps |
| `--learning-rate` | 0.01 | Adam learning rate |
| `--tol` | 1e-3 | Gradient norm convergence tolerance |
| `--update-interval` | 10 | Show interim stats every N simulations |
| `--seed` | 42 | Random seed for reproducibility |

## Usage Examples

### Quick test run (20 simulations, updates every 5)
```bash
./run_gpu_simulation_interactive.sh --n-sim 20 --update-interval 5
```

### Full run with frequent updates
```bash
./run_gpu_simulation_interactive.sh --n-sim 500 --update-interval 25
```

### High-quality run with more optimization steps
```bash
./run_gpu_simulation_interactive.sh --n-sim 1000 --max-steps 2000 --update-interval 50
```

## Monitoring Tips

### Good Recovery Signs
- **rho** (κ/λ ratio): r > 0.6 is excellent
- **kappa**: r > 0.3 is acceptable
- **lambda**: r > 0.2 is acceptable
- Correlations should stabilize after ~50 simulations

### When to Stop Early
- If rho correlation < 0.3 after 50 simulations
- If all individual parameter correlations are negative
- If you see NaN values appearing

### Interpreting Results

**r (correlation)**: How well estimated values correlate with true values
- r > 0.7: Excellent recovery
- r > 0.5: Good recovery  
- r > 0.3: Acceptable recovery
- r < 0.3: Poor recovery

**Bias**: Mean difference between estimated and true values
- Small absolute values are better

**RelBias%**: Bias as percentage of parameter range
- < 10%: Excellent
- < 20%: Good
- \> 30%: Concerning

## Output Files

Results are saved to: `simulations/results_v4_gpu_interactive.txt`

## Stopping the Simulation

Press **Ctrl+C** to stop. Partial results will not be saved, but you can run again with adjusted parameters.

## Comparison: Interactive vs Batch Mode

| Feature | Interactive | Batch (`run_gpu_simulation.sh`) |
|---------|-------------|----------------------------------|
| Progress bars | ✓ | Minimal |
| Interim stats | ✓ Every N sims | Only at 10% intervals |
| Real-time monitoring | ✓ Full | Limited |
| Speed | Same | Same |
| Use case | Monitoring/tuning | Production runs |

## Troubleshooting

**Issue**: Progress seems stuck  
**Solution**: Wait for GPU compilation on first run (1-2 mins), subsequent runs are faster

**Issue**: All correlations are negative  
**Solution**: This may indicate model issues. Try adjusting priors or design parameters.

**Issue**: Out of memory errors  
**Solution**: Reduce `--n-participants` or `--max-steps`
