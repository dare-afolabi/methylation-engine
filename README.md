# Differential Methylation Analysis Pipeline

A Python pipeline for genome-wide DNA methylation analysis using empirical Bayes moderated statistics.

<div align="center">
  <a href="https://github.com/dare-afolabi/methylation-engine/actions/workflows/ci.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/dare-afolabi/methylation-engine/ci.yml?branch=main&style=flat" alt="Build Status">
  </a>
  <a href="https://codecov.io/gh/dare-afolabi/methylation-engine">
    <img src="https://img.shields.io/codecov/c/github/dare-afolabi/methylation-engine?style=flat" alt="Coverage">
  </a>
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python 3.8+">
  </a>
  <a href="https://github.com/sponsors/dare-afolabi">
    <img src="https://img.shields.io/badge/Sponsor-lightgrey?style=flat&logo=github-sponsors" alt="Sponsor">
  </a>
</div>

## Overview

This pipeline implements limma-style empirical Bayes shrinkage for differential DNA methylation analysis, optimized for large-scale methylation arrays (450K, EPIC) and whole-genome bisulfite sequencing (WGBS).

### Key Features

- **Empirical Bayes variance shrinkage** (Smyth’s method)
- **Memory-efficient chunked processing** for 450K/EPIC arrays
- **Robust missing data handling** with per-CpG model fitting
- **Batch effect adjustment** via design matrix
- **Comprehensive diagnostics** and publication-quality plots
- **100x faster** than auto shrinkage with fixed prior option

-----

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/dare-afolabi/methylation-engine.git
cd methylation-engine

# Install dependencies
pip install .
```

### Basic Usage

```python
import pandas as pd
import numpy as np
from core.engine import fit_differential

# Load your M-values (CpGs × samples)
M = pd.read_csv("M_values.csv", index_col=0)

# Create design matrix
design = pd.DataFrame({
    "Intercept": 1,
    "Tumor": [0, 0, 0, 1, 1, 1]  # 0=Normal, 1=Tumor
}, index=M.columns)

# Run differential analysis
results = fit_differential(
    M=M,
    design=design,
    contrast=np.array([0, 1]),  # Test Tumor effect
    shrink='auto',
    robust=True
)

# Extract significant CpGs
sig_cpgs = results[results['padj'] < 0.05]
print(f"Found {len(sig_cpgs)} significant CpGs")
```

-----

## Input Requirements

### 1. M-values Matrix

- **Format**: Pandas DataFrame (CpGs × samples)
- **Values**: M-values (log2 ratio of methylated/unmethylated)
  - M = log2(β / (1-β)) where β = methylation proportion
- **Missing data**: Supported (NaN values handled per-CpG)
- **Scale**: Typically ranges from -5 to +5

**Example**:

```
             Sample1  Sample2  Sample3  Sample4
cg00000029   -0.523    0.234   -0.123    1.234
cg00000108    2.145    1.876    1.954    3.234
cg00000109   -1.234   -1.456   -1.123    0.234
```

### 2. Design Matrix

- **Format**: Pandas DataFrame (samples × covariates)
- **Index**: Must match M-values columns exactly
- **Columns**: Intercept + covariates (group, batch, etc.)
- **Values**: Numeric (0/1 for binary factors)

**Example**:

```python
design = pd.DataFrame({
    "Intercept": 1,
    "Tumor": [0, 0, 0, 1, 1, 1],
    "Batch": [0, 1, 0, 1, 0, 1]
}, index=M.columns)
```

### 3. Contrast Vector

- **Format**: NumPy array (length = design columns)
- **Purpose**: Specifies which coefficient(s) to test

**Examples**:

```python
# Test tumor effect (adjusting for batch)
contrast = np.array([0, 1, 0])  # [Intercept, Tumor, Batch]

# Test batch effect
contrast = np.array([0, 0, 1])
```

-----

## Pipeline Workflow

### 1. Data Preprocessing

```python
from core.engine import (
    filter_cpgs_by_missingness,
    impute_missing_values_fast,
    filter_min_per_group
)

# Filter CpGs with >20% missing
M_filtered, n_removed, n_kept = filter_cpgs_by_missingness(
    M,
    max_missing_rate=0.2,
    min_samples_per_group=5,
    groups=metadata['Type']
)

# Impute remaining missing values
M_imputed = impute_missing_values_fast(
    M_filtered,
    method='knn',
    k=5
)

# Ensure minimum samples per group
M_ready = filter_min_per_group(
    M_imputed,
    groups=metadata['Type'],
    min_per_group=5
)
```

### 2. Differential Analysis

#### Standard Method (< 50K CpGs)

```python
results = fit_differential(
    M=M_ready,
    design=design,
    contrast=np.array([0, 1]),
    shrink='auto',  # or 'smyth', 'median', 10.0
    robust=True
)
```

#### Chunked Method (450K/EPIC Arrays)

```python
from core.engine import fit_differential_chunked

results = fit_differential_chunked(
    M=M_large,
    design=design,
    chunk_size=10000,  # Process 10K CpGs at a time
    contrast=np.array([0, 1]),
    shrink=10.0,  # Fixed shrinkage for speed
    robust=True,
    verbose=True  # Show progress
)
```

### 3. Results Interpretation

```python
# View top hits
print(results[['logFC', 't', 'pval', 'padj']].head(10))

# Extract significant CpGs
from core.engine import get_significant_cpgs

sig_summary = get_significant_cpgs(
    results,
    lfc_thresh=1.5,
    pval_thresh=0.05,
    direction='hyper',  # or 'hypo', None
    return_summary=True
)

print(f"Hypermethylated: {sig_summary['n_hyper']}")
print(f"Hypomethylated: {sig_summary['n_hypo']}")
```

### 4. Visualization

```python
from core.engine import (
    plot_volcano_enhanced,
    plot_mean_variance,
    plot_sample_qc
)

# Volcano plot
plot_volcano_enhanced(
    results,
    lfc_thresh=1.5,
    pval_thresh=0.05,
    top_n=10,
    save_path="volcano.png"
)

# Diagnostic plots
plot_mean_variance(results, save_path="variance_shrinkage.png")
plot_sample_qc(M, metadata, group_col='Type', save_path="sample_qc.png")
```

-----

## Advanced Usage

### Batch Effect Adjustment

```python
# Design with batch covariate
design = pd.DataFrame({
    "Intercept": 1,
    "Tumor": [0, 0, 0, 1, 1, 1],
    "Batch": [0, 1, 0, 1, 0, 1]
}, index=M.columns)

# Test tumor effect (adjusted for batch)
contrast_tumor = np.array([0, 1, 0])

results = fit_differential(
    M, design,
    contrast=contrast_tumor,
    shrink='auto'
)
```

### Multi-Coefficient F-test

```python
# Test if tumor OR batch has effect
R = np.array([
    [0, 1, 0],  # Tumor
    [0, 0, 1]   # Batch
])

results_F = fit_differential(
    M, design,
    contrast_matrix=R,
    shrink='auto'
)

# Note: Use single contrasts for effect size interpretation
```

### Custom Shrinkage

```python
# Auto shrinkage (Smyth method)
results = fit_differential(M, design, contrast=c, shrink='auto')

# Fixed shrinkage (100x faster)
results = fit_differential(M, design, contrast=c, shrink=10.0)

# Median shrinkage
results = fit_differential(M, design, contrast=c, shrink='median')

# No shrinkage
results = fit_differential(M, design, contrast=c, shrink='none')
```

-----

## Performance Benchmarks

**Hardware**: 16GB RAM, 8-core CPU

|Dataset   |CpGs|Samples|Method  |Time  |Rate  |
|----------|----|-------|--------|------|------|
|Small     |10K |20     |Standard|2s    |5K/s  |
|Medium    |100K|20     |Standard|25s   |4K/s  |
|450K Array|450K|20     |Chunked |3min  |2.5K/s|
|EPIC Array|850K|20     |Chunked |6min  |2.4K/s|

**Optimization Tips**:

- Use `shrink=10.0` for 100x speedup (minimal accuracy loss)
- Increase `chunk_size` to 50K+ on high-RAM systems
- Filter low-variance CpGs before analysis

-----

## Output Columns

### Standard Differential Analysis

- **logFC**: Log2 fold change (M-value difference)
- **se**: Standard error (raw variance)
- **t**: Moderated t-statistic
- **pval**: Raw p-value
- **padj**: FDR-adjusted p-value (Benjamini-Hochberg)
- **df_resid**: Residual degrees of freedom
- **df_total**: Total degrees of freedom (df_resid + d0)
- **s2**: Raw variance estimate
- **s2_post**: Moderated (shrunken) variance
- **d0**: Prior degrees of freedom
- **n_obs**: Number of samples with data
- **meanM_GroupX**: Mean M-value per group
- **meanB_GroupX**: Mean beta-value per group (0-1)

### F-test Analysis

- **F**: F-statistic
- **pval/padj**: P-values
- **df1**: Numerator df (number of tested coefficients)
- **df2**: Denominator df (df_total)

-----

## Troubleshooting

### Issue: “Residual degrees of freedom <= 0”

**Cause**: Too few samples or too many covariates

**Fix**: Ensure n sample > n covariates + 1

### Issue: High memory usage with large arrays

**Cause**: Processing too many CpGs at once

**Fix**: Reduce `chunk_size` in `fit_differential_chunked()`

### Issue: “No finite variances provided”

**Cause**: All CpGs filtered out or have zero variance

**Fix**: Check filtering thresholds and data quality

### Issue: Low sensitivity (missing true positives)

**Cause**: Over-shrinkage or insufficient effect size

**Fix**:

- Use `shrink='none'` to check raw results
- Lower `lfc_thresh` (e.g., 1.0 instead of 1.5)
- Increase sample size

### Issue: Inflated p-values (Q-Q plot deviation)

**Cause**: Batch effects, outliers, or model misspecification

**Fix**:

- Add batch covariates to design matrix
- Use `robust=True` for winsorization
- Check sample QC plots

-----

## Citation

If you use this pipeline, please cite:

```bibtex
@software{methylation_engine,
  author = {Dare Afolabi},
  title = {Differential Methylation Analysis Pipeline},
  year = {2025},
  url = {https://github.com/dare-afolabi/methylation-engine}
}
```

**Key References**:

- Smyth, G. K. (2004). Linear models and empirical bayes methods for assessing differential expression in microarray experiments. *Statistical Applications in Genetics and Molecular Biology*, 3(1).
- Phipson, B. et al. (2016). missMethyl: an R package for analyzing data from Illumina’s HumanMethylation450 platform. *Bioinformatics*, 32(2), 286-288.

-----

## License

MIT License - see [LICENSE](https://github.com/dare-afolabi/methylation-engine/blob/main/LICENSE) file for details

-----

## Support

- **Issues**: [https://github.com/dare-afolabi/methylation-engine/issues](https://github.com/dare-afolabi/methylation-engine/issues)
- **Discussions**: [https://github.com/discussions](https://github.com/discussions)
- **Email**: [dare.afolabi@outlook.com](mailto:dare.afolabi@outlook.com)

-----

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

See [CONTRIBUTING.md](https://github.com/dare-afolabi/methylation-engine/blob/main/docs/CONTRIBUTING.md) for details.