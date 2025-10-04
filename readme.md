# TIC (The Information Clock)

A dual-process mathematical framework for prospective time perception through information dynamics.

## Abstract

Time perception compresses under sustained information load while dilating under transient novelty. TIC formalizes this through competing mechanisms: sustained information density drives temporal compression ("time flies when busy") while transient novelty drives temporal dilation (oddball effect). The framework quantifies how information processing speed determines subjective duration.

## Core Model

```
T_s ≈ T_o · [1 + κ·N'^γ] / [λ · D' · Φ']
```

Where:
- `T_s/T_o` = subjective/objective duration ratio
- `D'` = normalized information density (compression factor, denominator)
- `N'` = normalized novelty (dilation factor, numerator)
- `Φ'` = individual integrative capacity (trait-level processing efficiency)
- `λ, κ, γ` = empirically determined parameters (primary model; α=β=1)

Exploratory extension (Appendix B): allow exponents on D' and Φ' (α, β) to vary.

## Key Findings

Parameter recovery validation (N=35, 1960 observations):
- **Compression parameters (primary)**: r ≈ 0.80 (good), supporting identifiability under the simplified 3-parameter model {λ, κ, γ} with α=β=1
- **Dilation parameters (exploratory)**: r ≈ 0.50 (moderate), suitable for exploratory analysis and future designs with more novelty levels
- Mean recovery: r ≈ 0.68 across all parameters

The asymmetric recovery supports TIC's core innovation—the compression mechanism—while acknowledging that transient novelty effects require expanded experimental trials.

## Repository Structure

```
TIC/
├── manuscript/       # Main paper and theoretical framework
│   └── TIC_manuscript.md
├── figures/          # Visual diagrams and illustrations
│   ├── figure1.html  # Interactive Figure 1 (dual-process model)
│   ├── figure-1.png  # Complete figure
│   └── panel-*.png   # Individual panels
├── simulations/      # Parameter recovery validation
│   ├── parameter_recovery.py
│   └── results.txt
├── scripts/          # Build and utility scripts
│   └── make_pdf.sh
└── readme.md
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

## Reproducibility

- The parameter recovery simulation (`simulations/parameter_recovery.py`) uses a fixed random seed for deterministic results.
- Running the script writes a human-readable summary to `simulations/results.txt` and prints a Markdown table of recovery metrics.

## Development

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
- 10 bits/s consciousness bottleneck (Zheng & Meister, 2024)
- Perturbational Complexity Index (Casarotto et al., 2016)

## Limitations

- Addresses prospective timing only (experience during events)
- Retrospective timing (memory of duration) shows opposite patterns
- Novelty parameters require expanded trials for precise estimation
- Φ' is modeled as a stable trait by design, deliberately excluding state-dependent fluctuations in this foundational version

## License

MIT
