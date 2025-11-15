# Analysis Guide

Complete workflow for differential DNA methylation analysis

## Quick Start

```python
import pandas as pd
import numpy as np
from methylation_engine.core.engine import fit_differential

# Load data
M = pd.read_csv("M_values.csv", index_col=0)

# Design matrix
design = pd.DataFrame({
    "Intercept": 1,
    "Tumor": [0]*10 + [1]*10
}, index=M.columns)

# Analyze
results = fit_differential(
    M, design,
    contrast=np.array([0, 1]),
    shrink='auto'
)

# Extract significant CpGs
sig = results[results['padj'] < 0.05]
```

## Input Requirements

### M-values Matrix

- **Format**: DataFrame (CpGs × samples)
- **Values**: log2(β / (1-β))
- **Range**: Typically -5 to +5
- **Missing data**: Supported (NaN)

**Convert from β-values**:

```python
M = np.log2(beta / (1 - beta))
```

### Design Matrix

- **Format**: DataFrame (samples × covariates)
- **Index**: Must match M columns
- **Values**: Numeric (0/1 for factors)

```python
design = pd.DataFrame({
    "Intercept": 1,
    "Group": [0, 0, 0, 1, 1, 1],
    "Batch": [0, 1, 0, 1, 0, 1]
}, index=M.columns)
```

### Contrast Vector

- **Format**: NumPy array
- **Length**: Must match design columns

```python
contrast = np.array([0, 1, 0])  # Test Group, adjust for Batch
```

## Preprocessing

### 1. Filter by Missingness

```python
from methylation_engine.core.engine import filter_cpgs_by_missingness

M_filtered, n_removed, n_kept = filter_cpgs_by_missingness(
    M,
    max_missing_rate=0.2,
    min_samples_per_group=5,
    groups=metadata['Type']
)
```

### 2. Impute Missing Values

```python
from methylation_engine.core.engine import impute_missing_values_fast

# Fast methods
M_imputed = impute_missing_values_fast(M_filtered, method='median')
M_imputed = impute_missing_values_fast(M_filtered, method='mean')

# KNN (slower, better for small datasets)
M_imputed = impute_missing_values_fast(
    M_filtered,
    method='knn',
    k=5,
    use_sample_knn=True  # Faster if samples < CpGs
)
```

### 3. Filter Per-Group Counts

```python
from methylation_engine.core.engine import filter_min_per_group

M_ready = filter_min_per_group(
    M_imputed,
    groups=metadata['Type'],
    min_per_group=5,
    verbose=True
)
```

## Differential Analysis

### Standard Method (< 50K CpGs)

```python
from methylation_engine.core.engine import fit_differential

results = fit_differential(
    M=M_ready,
    design=design,
    contrast=np.array([0, 1]),
    shrink='auto',  # or 'smyth', 'median', 10.0, 'none'
    robust=True,
    return_residuals=False
)
```

### Chunked Method (450K/EPIC Arrays)

```python
from methylation_engine.core.engine import fit_differential_chunked

results = fit_differential_chunked(
    M=M_large,
    design=design,
    chunk_size=10000,
    contrast=np.array([0, 1]),
    shrink=10.0,  # Fixed shrinkage for speed
    robust=True,
    verbose=True
)
```

## Shrinkage Options

|Method    |Speed    |Accuracy|Use Case      |
|----------|---------|--------|--------------|
|`'auto'`  |Slow     |Best    |First analysis|
|`'smyth'` |Slow     |Best    |Publication   |
|`'median'`|Fast     |Good    |Quick checks  |
|`10.0`    |Very Fast|Good    |Large datasets|
|`'none'`  |Fast     |Poor    |Debugging only|

**Recommendation**: Use `shrink=10.0` for 100x speedup with <1% accuracy loss.

## Results Interpretation

### Output Columns

|Column    |Description             |
|----------|------------------------|
|`logFC`   |Log2 fold change (ΔM)   |
|`se`      |Standard error          |
|`t`       |Moderated t-statistic   |
|`pval`    |Raw p-value             |
|`padj`    |FDR-adjusted p-value    |
|`df_total`|Total degrees of freedom|
|`s2`      |Raw variance            |
|`s2_post` |Moderated variance      |
|`d0`      |Prior degrees of freedom|
|`meanM_*` |Mean M-value per group  |
|`meanB_*` |Mean β-value per group  |

### Extract Significant CpGs

```python
from methylation_engine.core.engine import get_significant_cpgs

# All significant
sig_summary = get_significant_cpgs(
    results,
    lfc_thresh=1.5,
    pval_thresh=0.05,
    return_summary=True
)

# Hypermethylated only
hyper = get_significant_cpgs(
    results,
    direction='hyper',
    lfc_thresh=1.5,
    pval_thresh=0.05
)

# With delta-beta threshold
sig_strong = get_significant_cpgs(
    results,
    delta_beta_thresh=0.10,
    pval_thresh=0.05
)
```

### Summary Statistics

```python
from methylation_engine.core.engine import summarize_differential_results

summary = summarize_differential_results(results, pval_thresh=0.05)

print(f"Significant: {summary['significant']:,}")
print(f"Hyper: {summary['hypermethylated']:,}")
print(f"Hypo: {summary['hypomethylated']:,}")
print(f"Shrinkage factor: {summary['shrinkage_factor']:.2f}x")
```

## Batch Effect Adjustment

### Single Contrast (Recommended)

```python
design = pd.DataFrame({
    "Intercept": 1,
    "Tumor": [0]*3 + [1]*3,
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
```

**Note**: Use single contrasts for effect size interpretation. F-tests are for QC only.

## Visualization

### Volcano Plot

```python
from methylation_engine.core.engine import plot_volcano_enhanced

plot_volcano_enhanced(
    results,
    lfc_thresh=1.5,
    pval_thresh=0.05,
    top_n=10,
    save_path='volcano.png'
)
```

### Diagnostic Plots

```python
from methylation_engine.core.engine import (
    plot_mean_variance,
    plot_pvalue_qq,
    plot_residual_diagnostics
)

# Variance shrinkage
plot_mean_variance(results, save_path='variance.png')

# P-value inflation check
plot_pvalue_qq(results, save_path='qq.png')

# Residual normality
results, residuals = fit_differential(
    M, design,
    contrast=contrast,
    return_residuals=True
)
plot_residual_diagnostics(
    residuals, M, design,
    top_n=9,
    save_path='residuals.png'
)
```

### Sample QC

```python
from methylation_engine.core.engine import plot_sample_qc

plot_sample_qc(
    M, metadata,
    group_col='Type',
    save_path='sample_qc.png'
)
```

## Export Results

```python
from methylation_engine.core.engine import export_results

# Significant only
sig = results[results['padj'] < 0.05]
export_results(sig, 'significant.csv', format='csv')

# All results
export_results(results, 'all_results.xlsx', format='excel')
```

## Performance Tips

### Speed Optimization

1. **Use fixed shrinkage**: `shrink=10.0` (100x faster)
2. **Increase chunk size**: `chunk_size=50000` (if RAM allows)
3. **Filter low-variance CpGs**: Filter before analysis
4. **Use median imputation**: Faster than KNN

### Memory Optimization

1. **Decrease chunk size**: `chunk_size=5000`
2. **Process in batches**: Split large datasets
3. **Use generators**: For very large files
4. **Clear intermediate results**: delete unused DataFrames

## Troubleshooting

### “Residual degrees of freedom <= 0”

**Fix**: Ensure n samples > n covariates + 1

### High memory usage

**Fix**: Reduce `chunk_size` in chunked analysis

### “No finite variances”

**Fix**: Check filtering thresholds and data quality

### Low sensitivity

**Fix**:

- Use `shrink='none'` to check
- Lower `lfc_thresh`
- Increase sample size

### Inflated p-values

**Fix**:

- Add batch covariates
- Use `robust=True`
- Check sample QC

See [Troubleshooting](https://github.com/dare-afolabi/methylation-engine/blob/main/docs/TROUBLESHOOTING.md) for more details

## Examples

See `demos/engine_demo.py` for:

- Complete preprocessing workflow
- Standard and chunked analysis
- Batch effect adjustment
- Diagnostic visualization
- Export and reporting

## References

- Smyth, G. K. (2004). Linear models and empirical bayes methods for assessing differential expression in microarray experiments. *Statistical Applications in Genetics and Molecular Biology*, 3(1).
- Phipson, B. et al. (2016). missMethyl: an R package for analyzing data from Illumina’s HumanMethylation450 platform. *Bioinformatics*, 32(2), 286-288.