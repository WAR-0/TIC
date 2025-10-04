# TIC Parameter Recovery v5 (Final)

## Overview

Version 5 represents a complete redesign with **evidence-aligned priors** and improved noise modeling based on theoretical considerations.

## Key Improvements Over v4

### 1. Evidence-Aligned Priors
- **Broader bounds**: D₀ ∈ [0.02, 0.40] (was [0.05, 0.20])
- **Weaker penalties**: Weight=40 for D₀ (was 80), Weight=8 for others (was 10)
- **Updated means**: Reflect literature uncertainty more accurately

### 2. Ex-Gaussian Noise Model
- Combines **Gaussian** base noise + **Exponential** tail for attention lapses
- More realistic than Student-t for modeling asymmetric response distributions
- σ = 4% of T₀, τ = 10% of T₀

### 3. Simplified Design
- **No calibration block** (was causing identifiability issues)
- **3 blocks**: Density manipulation, Novelty manipulation, Low-density trials
- **69 trials per participant** (down from 84)

### 4. Pure NumPy Implementation
- Custom Adam optimizer with manual gradient computation
- No JAX dependency for v5 (more portable)
- Logistic reparameterization for bounded parameters

## Running v5

### Quick Start

```bash
# Default: 200 simulations, 35 participants
./run_v5_simulation.sh

# Custom parameters
./run_v5_simulation.sh --n-sim 500 --n-participants 35

# Using environment variables
N_SIM=1000 N_PARTICIPANTS=50 python3 simulations/parameter_recovery_v5_final.py
```

### Expected Runtime
- **3 simulations**: ~30 seconds
- **200 simulations**: ~30-40 minutes
- **1000 simulations**: ~3 hours

## Results Interpretation

### Test Run (3 simulations)

```
Parameter           r       Bias   RelBias%       RMSE        MAE
------------------------------------------------------------------------
D0          -0.718     0.199      52.4     0.201     0.199
lambda       0.826    -0.259     -10.0     0.296     0.272
kappa        0.568    -0.018      -0.9     0.105     0.083
gamma       -0.001    -0.059      -3.3     0.227     0.169
rho          0.660     0.050       1.0     0.129     0.088
```

**Key metrics** (3 sims only - need more for stability):
- **lambda**: r = 0.826 ✓ (excellent recovery)
- **rho** (κ/λ): r = 0.660 ✓ (good recovery)
- **kappa**: r = 0.568 (acceptable)
- **D0**: r = -0.718 (needs investigation with more sims)

## Design Details

### Block A: Density Manipulation (30 trials)
- 5 density levels: [0.125, 0.3125, 0.5, 0.6875, 0.875]
- 6 trials per level
- Novelty randomized within [0.2, 0.8]

### Block B: Novelty Manipulation (24 trials)
- 4 novelty levels: [0.2, 0.4, 0.6, 0.8]
- 6 trials per level
- Fixed density = 0.25

### Block C: Low-Density Trials (15 trials)
- 3 density levels: [0.05, 0.15, 0.30]
- 5 trials per level
- Low novelty (mean=0.12, sd=0.04)
- Helps constrain D₀ estimation

## Parameter Bounds

| Parameter | Lower | Upper | Prior Mean | Prior Scale |
|-----------|-------|-------|------------|-------------|
| D₀        | 0.02  | 0.40  | 0.15       | 0.08        |
| λ         | 0.4   | 3.0   | 1.0        | 0.4         |
| κ         | 0.05  | 2.0   | 0.2        | 0.5         |
| γ         | 0.2   | 2.0   | 0.8        | 0.4         |

## Output Files

Results saved to: `simulations/results_v5.txt`

## Comparison with v4

| Feature | v4 | v5 |
|---------|----|----|
| Noise model | Student-t | Ex-Gaussian |
| Calibration block | Yes (30 trials) | No |
| Trials/participant | 84 | 69 |
| D₀ bounds | [0.05, 0.20] | [0.02, 0.40] |
| Prior strength | Strong (80/10) | Weaker (40/8) |
| Implementation | JAX | Pure NumPy |
| Optimizer | Optax Adam | Custom Adam |

## Theoretical Justification

### Why Ex-Gaussian?
- Models **base variability** (Gaussian component)
- Models **attention lapses** (Exponential tail)
- Empirically observed in RT distributions
- More principled than symmetric Student-t

### Why No Calibration?
- Zero-novelty trials create **artificial constraint** on D₀
- May force unrealistic parameter trade-offs
- Real experiments won't have perfect zero-novelty trials
- Low-density block provides sufficient D₀ constraint

### Why Broader D₀ Prior?
- Literature estimates vary widely (0.05-0.30)
- Overly tight priors may bias recovery
- Let data speak more strongly

## Next Steps

1. **Run full validation**: `./run_v5_simulation.sh --n-sim 1000`
2. **Compare with v4**: Check if rho recovery improves
3. **Assess D₀ recovery**: Broader prior may help or hurt
4. **Document final results**: Update manuscript with v5 findings

## Technical Notes

- Uses **Huber loss** (δ=1.0) for robustness to outliers
- **Hierarchical**: Shared D₀, individual λ/κ/γ per participant
- **Convergence**: tol=1e-3 on gradient norm
- **Max iterations**: 2000 (typically converges ~600-1000)

