# Differential Methylation Analysis Pipeline

A Python toolkit for DNA methylation analysis and study planning

<div align="center">
  <a href="https://github.com/dare-afolabi/methylation-engine/actions/workflows/ci.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/dare-afolabi/methylation-engine/ci.yml?branch=main&style=flat" alt="Build Status">
  </a>
  <a href="https://codecov.io/gh/dare-afolabi/methylation-engine">
    <img src="https://img.shields.io/codecov/c/github/dare-afolabi/methylation-engine?style=flat" alt="Coverage">
  </a>
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+">
  </a>
  <a href="https://github.com/sponsors/dare-afolabi">
    <img src="https://img.shields.io/badge/Sponsor-grey?style=flat&logo=github-sponsors" alt="Sponsor">
  </a>
</div>

## Overview

Complete pipeline for DNA methylation research:
- **Study Planning** (`core.planner`) - Sample size, power analysis, cost estimation
- **Data Analysis** (`core.engine`) - Differential methylation with empirical Bayes
- **Quality Control** - Comprehensive diagnostics and visualization

## Quick Start

### Installation

```bash
git clone https://github.com/dare-afolabi/methylation-engine.git
cd methylation-engine
pip install .
```

### Study Planning (Before Data Collection)

```python
from methylation_engine.core.planner import create_study_plan

plan = create_study_plan(
    study_name="Cancer Methylation Study",
    expected_delta_beta=0.10,  # 10% methylation difference
    target_power=0.80,
    platform='EPIC',
    budget=25000,
    design_type='paired',
    output_dir='my_study_plan'
)

print(f"Recommended N: {plan['sample_size']['recommended']['n_per_group']}")
print(f"Estimated cost: ${plan['costs']['total']:,}")
```

### Data Analysis (After Data Collection)

```python
import pandas as pd
import numpy as np
from methylation_engine.core.engine import fit_differential

# Load M-values
M = pd.read_csv("M_values.csv", index_col=0)

# Create design matrix
design = pd.DataFrame({
    "Intercept": 1,
    "Tumor": [0]*10 + [1]*10
}, index=M.columns)

# Run analysis
results = fit_differential(
    M, design,
    contrast=np.array([0, 1]),
    shrink='auto'
)

sig = results[results['padj'] < 0.05]
print(f"Found {len(sig)} significant CpGs")
```

## Documentation

- [Study Planning Guide](https://github.com/dare-afolabi/methylation-engine/blob/main/docs/PLANNING.md) - Sample size, costs, timelines
- [Analysis Guide](https://github.com/dare-afolabi/methylation-engine/blob/main/docs/ANALYSIS.md) - Differential methylation workflow
- [Troubleshooting](https://github.com/dare-afolabi/methylation-engine/blob/main/docs/TROUBLESHOOTING.md) - Common issues and solutions
- [Contributing](https://github.com/dare-afolabi/methylation-engine/blob/main/docs/CONTRIBUTING.md) (and [Code of Conduct](https://github.com/dare-afolabi/methylation-engine/blob/main/docs/CODE_OF_CONDUCT.md)) - Development guidelines

## Key Features

### Study Planner (`core.planner`)

- Power analysis with empirical Bayes assumptions
- Sample size recommendations (minimum/recommended/optimal)
- Cost estimation by platform (450K, EPIC, WGBS, etc.)
- Timeline projection with phase breakdown
- Batch planning and risk assessment
- Platform/design comparisons

### Analysis Engine (`core.engine`)
- Empirical Bayes variance shrinkage (Smyth’s method)
- Memory-efficient chunked processing (450K/EPIC arrays)
- Robust missing data handling
- Batch effect adjustment via design matrix
- Comprehensive diagnostics and visualization
- 100x faster with fixed shrinkage option

## Quick Function Reference

### Planning Functions

```python
from methylation_engine.core.planner import (
    plan_sample_size,      # Calculate recommended sample sizes
    estimate_costs,         # Estimate study costs
    estimate_timeline,      # Project study timeline
    compare_platforms,      # Compare platform costs/specs
    compare_designs,        # Compare design requirements
    create_study_plan       # Generate complete plan
)
```

### Analysis Functions

```python
from methylation_engine.core.engine import (
    fit_differential,            # Standard differential analysis
    fit_differential_chunked,    # Memory-efficient chunked analysis
    filter_cpgs_by_missingness,  # Filter by missing data
    impute_missing_values_fast,  # Impute missing values
    get_significant_cpgs,        # Extract significant results
    plot_volcano_enhanced,       # Volcano plot
    plot_sample_qc              # Sample QC visualization
)
```

See demos for complete sample workflows:
- **Planning**: `demos/planner_demo.py`
- **Analysis**: `demos/engine_demo.py`

## Performance
- **Study Planning**: < placeholder for complete workflow including visualization

- **Analysis Benchmarks (16GB RAM, 4-core CPU)**:

| Dataset | CpGs | Samples | Time | Rate |
|---------|------|---------|------|------|
| Small | 10K | 24 | placeholder | 5K/s |
| Medium | 100K | 24 | placeholder | 4K/s |
| 450K Array | 450K | 24 | placeholder | 3.2K/s |
| EPIC Array | 850K | 24 | placeholder | 2.4K/s |

*See demo outputs for timings on your hardware*

## Examples

### Complete Study Planning

```python
from methylation_engine.core.planner import plan_sample_size, compare_platforms

# Sample size for paired design
result = plan_sample_size(
    expected_delta_beta=0.10,
    target_power=0.80,
    design_type='paired'
)
print(f"Need {result['recommended']['n_per_group']} per group")

# Compare platform options
comparison = compare_platforms(
    n_samples=24,
    platforms=['EPIC', 'EPICv2', 'WGBS']
)
print(comparison)
```

### Chunked Analysis for Large Arrays

```python
from methylation_engine.core.engine import fit_differential_chunked

results = fit_differential_chunked(
    M=M_450k,  # 450,000 CpGs
    design=design,
    chunk_size=10000,
    contrast=np.array([0, 1]),
    shrink=10.0,  # Fixed for speed
    verbose=True
)
```

### Batch Effect Adjustment

```python
# Design with batch covariate
design = pd.DataFrame({
    "Intercept": 1,
    "Tumor": [0, 0, 0, 1, 1, 1],
    "Batch": [0, 1, 0, 1, 0, 1]
}, index=M.columns)

# Test tumor effect (adjusted for batch)
results = fit_differential(
    M, design,
    contrast=np.array([0, 1, 0]),  # Test Tumor, adjust for Batch
    shrink='auto'
)
```

## Module Structure

```bash
methylation-engine/
├── core/
│   ├── config.py       # Platform/design database
│   ├── planner.py      # Study planning
│   └── engine.py       # Statistical analysis
├── demos/
│   ├── planner_demo.py # Planning workflow
│   └── engine_demo.py  # Analysis workflow
└── tests/
    ├── test_planner.py
    └── test_engine.py
```

## Citation

```bibtex
@software{methylation_engine,
  author = {Dare Afolabi},
  title = {Differential Methylation Analysis Pipeline},
  year = {2025},
  url = {https://github.com/dare-afolabi/methylation-engine}
}
```

### References

- Smyth, G. K. (2004). Linear models and empirical bayes methods for assessing differential expression in microarray experiments. *Statistical Applications in Genetics and Molecular Biology*, 3(1).
- Liu, P., & Hwang, J.T.G. (2007). Quick calculation for sample size while controlling false discovery rate with application to microarray analysis. *Bioinformatics*, 23(6), 739–746.
- Du, P., Zhang, X., Huang, C.-C., Jafari, N., Kibbe, W.A., Hou, L., & Lin, S. (2010). Comparison of Beta-value and M-value methods for quantifying methylation levels by microarray analysis. *BMC Bioinformatics*, 11:587.
- Jung, S.H., Young, S.S. (2012). Power and sample size calculation for microarray studies. *Journal of Biopharmaceutical Statistics*, 22(1):30-42.
- Phipson, B. et al. (2016). missMethyl: an R package for analyzing data from Illumina’s HumanMethylation450 platform. *Bioinformatics*, 32(2), 286-288.

## Support

- **Issues**: [GitHub Issues](https://github.com/dare-afolabi/methylation-engine/issues)
- **Discussions**: [GitHub Discussions](https://github.com/dare-afolabi/methylation-engine/discussions/4)
- **Email**: [dare.afolabi@outlook.com](mailto:dare.afolabi@outlook.com)

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

See [CONTRIBUTING.md](https://github.com/dare-afolabi/methylation-engine/blob/main/docs/CONTRIBUTING.md) for details

## License

MIT License - see [LICENSE](https://github.com/dare-afolabi/methylation-engine/blob/main/LICENSE) file for details