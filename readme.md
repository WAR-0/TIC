# TIC (The Information Clock)

A dual-process mathematical framework for prospective time perception through information dynamics.

## Abstract

Time perception compresses under sustained information load while dilating under transient novelty. TIC formalizes this through competing mechanisms: sustained information density drives temporal compression ("time flies when busy") while transient novelty drives temporal dilation (oddball effect). The framework quantifies how information processing speed determines subjective duration.

## Core Model

```
T_s ≈ T_o · [1 + κ·N'^γ] / [λ(D')^α · Φ'^β]
```

Where:
- `T_s/T_o` = subjective/objective duration ratio
- `D'` = normalized information density (compression factor, denominator)
- `N'` = normalized novelty (dilation factor, numerator)  
- `Φ'` = individual integrative capacity (trait-level processing efficiency)
- `λ, κ, α, β, γ` = empirically determined parameters

## Key Findings

Parameter recovery validation (N=35, 1960 observations):
- **Compression parameters** (novel contribution): r > 0.79, validating information-density mechanism
- **Dilation parameters** (known effects): r ≈ 0.50, confirming established oddball phenomena
- Mean recovery: r = 0.677 across all parameters

The asymmetric recovery validates TIC's core innovation—the compression mechanism—while acknowledging that transient novelty effects require expanded experimental trials.

## Repository Structure

```
/manuscript/    # Theoretical framework and experimental design
/simulations/   # Parameter recovery and validation code
/figures/       # Conceptual diagrams
/analysis/      # Power analysis and statistical validation
```

## Theoretical Contributions

1. **Dual-process architecture**: Compression and dilation operate through separate mechanisms (denominator vs numerator)
2. **Information-theoretic quantification**: Shannon entropy (D') and KL divergence (N') as measurable predictors
3. **Individual differences**: Trait integrative capacity (Φ') explains why identical experiences feel longer/shorter to different people
4. **Falsifiable predictions**: Five specific hypotheses with quantitative effect size estimates

## Empirical Approach

- **Paradigm**: Rapid Serial Visual Presentation (RSVP) at 30Hz
- **Manipulation**: Shannon entropy (1.0-7.0 bits/frame) and sequence predictability
- **Neural markers**: Lempel-Ziv complexity (LZc) as Φ' proxy, validated against Perturbational Complexity Index (PCI)
- **Task**: Duration production (estimate 60s during 90s blocks)

## Development

Framework developed in 24 hours using human-AI collaborative synthesis across:
- Information theory (10 bits/s processing bottleneck)
- Consciousness research (Integrated Information Theory)
- Time perception literature (15+ years oddball studies)
- Bayesian brain theories (predictive processing)

## Status

- [x] Mathematical formalization complete
- [x] Parameter recovery validated
- [x] Power analysis conducted
- [ ] Empirical validation pending
- [ ] Preprint submission in progress

## Citation

```
[Forthcoming - arXiv preprint]
Repository: https://github.com/WAR-0/TIC
```

## Related Work

TIC builds on established findings:
- Oddball temporal dilation (Tse et al., 2004)
- Attentional gate models (Zakay & Block, 1995)
- 10 bits/s consciousness bottleneck (Zheng & Meister, 2025)
- Perturbational Complexity Index (Casarotto et al., 2016)

## Limitations

- Addresses prospective timing only (experience during events)
- Retrospective timing (memory of duration) shows opposite patterns
- Novelty parameters require expanded trials for precise estimation
- Individual Φ' treated as trait, not state

## License

MIT