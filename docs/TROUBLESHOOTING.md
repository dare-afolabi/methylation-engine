# Troubleshooting Guide

Common issues and solutions for both planning and analysis

## Study Planning Issues

### Sample size too large for budget

**Symptoms**: Recommended N exceeds available budget

**Solutions**:
```python
# 1. Switch to cheaper platform
compare_platforms(n_samples, platforms=['450K', 'EPIC'])

# 2. Use paired design (lower N required)
plan_sample_size(delta_beta=0.10, design_type='paired')

# 3. Accept lower power
plan_sample_size(delta_beta=0.10, target_power=0.70)

# 4. Focus on larger effects
plan_sample_size(delta_beta=0.15, target_power=0.80)
```

### Power too low with available N

**Symptoms**: Achieved power < 70%

**Solutions**:

- Increase sample size
- Use paired design
- Accept lower power with larger effect thresholds
- Phase study (collect more samples later)

### Platform recommendations unclear

**Symptoms**: Multiple platforms within budget

**Criteria**:

- **EPIC**: Best balance (867K CpGs, $500)
- **EPICv2**: Latest technology (935K CpGs, $550)
- **450K**: Budget option (485K CpGs, $400)
- **WGBS**: High resolution ($1200, overkill for most studies)

## Analysis Issues

### Memory Errors

**Symptoms**: `MemoryError` or system slowdown

**Solutions**:

```python
# 1. Use chunked analysis
fit_differential_chunked(M, design, chunk_size=5000)

# 2. Filter aggressively before analysis
M_filtered = filter_min_per_group(M, min_per_group=8)

# 3. Use median imputation (not KNN)
M_imputed = impute_missing_values_fast(M, method='median')

# 4. Process on high-RAM machine or cluster
```

### “Residual degrees of freedom <= 0”

**Symptoms**: ValueError during fitting

**Cause**: Too many covariates for sample size

**Solutions**:

```python
# Check: n_samples must be > n_covariates + 1
n_samples = design.shape[0]
n_covariates = design.shape[1]

if n_samples <= n_covariates:
    # Remove non-essential covariates
    design = design[['Intercept', 'Group']]
    
    # OR collect more samples
```

### “No finite variances provided”

**Symptoms**: Warning during shrinkage estimation

**Causes**:

- All CpGs filtered out
- Zero-variance CpGs
- Extreme outliers

**Solutions**:

```python
# 1. Check filtering thresholds
M_filtered, n_removed, n_kept = filter_cpgs_by_missingness(
    M, max_missing_rate=0.30  # Less strict
)

# 2. Check for zero-variance CpGs
var_per_cpg = M.var(axis=1)
M = M[var_per_cpg > 0.01]

# 3. Use robust=True
results = fit_differential(M, design, robust=True)
```

### Low sensitivity (missing true positives)

**Symptoms**: Known DM sites not detected

**Causes**:

- Over-shrinkage
- Insufficient power
- Batch effects

**Solutions**:

```python
# 1. Test without shrinkage
results_raw = fit_differential(M, design, shrink='none')

# 2. Lower thresholds
sig = results[results['padj'] < 0.10]  # Instead of 0.05

# 3. Add batch covariates
design['Batch'] = metadata['Batch']

# 4. Check power was adequate
from methylation_engine.core.planner import calculate_power
power = calculate_power(n_per_group=12, effect_size=1.5)
```

### Inflated p-values

**Symptoms**: Q-Q plot deviates from diagonal

**Causes​​​​​​​​​​​​​​​**:

- Batch effects
- Population stratification
- Technical artifacts
- Outlier samples

**Solutions**:

```python
# 1. Check sample QC
from methylation_engine.core.engine import plot_sample_qc
plot_sample_qc(M, metadata, group_col='Type')

# 2. Add batch to design
design['Batch'] = metadata['Batch']

# 3. Use robust shrinkage
results = fit_differential(M, design, robust=True)

# 4. Remove outlier samples
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
coords = pca.fit_transform(M.T)
# Identify and remove outliers manually

# 5. Check for technical artifacts
missing_per_sample = M.isna().sum(axis=0)
bad_samples = missing_per_sample[missing_per_sample > 0.15].index
M_clean = M.drop(columns=bad_samples)
```

### Slow performance

**Symptoms**: Analysis takes hours

**Solutions**:

```python
# 1. Use fixed shrinkage (100x faster)
results = fit_differential(M, design, shrink=10.0)

# 2. Use chunked method
results = fit_differential_chunked(
    M, design,
    chunk_size=50000,  # Increase if RAM allows
    shrink=10.0
)

# 3. Filter low-variance CpGs first
var_threshold = M.var(axis=1).quantile(0.25)
M_filtered = M[M.var(axis=1) > var_threshold]

# 4. Use faster imputation
M_imputed = impute_missing_values_fast(M, method='median')
```

### “Design index must match M columns”

**Symptoms**: ValueError during validation

**Cause**: Sample names don’t match

**Solutions**:

```python
# Check alignment
print(M.columns)
print(design.index)

# Fix alignment
design = design.loc[M.columns]

# OR subset M to match design
M = M[design.index]
```

### Missing group means in results

**Symptoms**: No `meanM_*` or `meanB_*` columns

**Cause**: Bug in chunked analysis (fixed in provided patches)

**Solutions**:

```python
# Use latest version with patches applied
results = fit_differential_chunked(M, design, ...)

# OR add manually
from methylation_engine.core.engine import _add_group_means
results = _add_group_means(results, M, design)
```

## Data Quality Issues

### High missingness

**Symptoms**: >10% missing values per sample

**Solutions**:

```python
# 1. Check per-sample missingness
missing_per_sample = M.isna().mean(axis=0) * 100
bad_samples = missing_per_sample[missing_per_sample > 15]

# 2. Remove bad samples
M_clean = M.drop(columns=bad_samples.index)

# 3. Use stricter CpG filtering
M_filtered, _, _ = filter_cpgs_by_missingness(
    M, max_missing_rate=0.10  # More strict
)
```

### Batch confounding

**Symptoms**: Batch correlates with group

**Solutions**:

```python
# 1. Check confounding
import pandas as pd
confound_table = pd.crosstab(
    metadata['Group'],
    metadata['Batch']
)
print(confound_table)

# 2. If fully confounded, cannot separate effects
# Need to re-design study or collect more samples

# 3. If partially confounded, include batch
design['Batch'] = metadata['Batch']
results = fit_differential(M, design, contrast=[0, 1, 0])
```

### Zero or low variance CpGs

**Symptoms**: Many CpGs with var ≈ 0

**Solutions**:

```python
# Remove zero-variance CpGs
M = M[M.var(axis=1) > 1e-6]

# OR filter by variance quantile
var_threshold = M.var(axis=1).quantile(0.10)
M = M[M.var(axis=1) > var_threshold]
```

## Interpretation Issues

### logFC and delta-beta don’t match

**Question**: Why doesn’t Δβ = 0.10 give logFC = 0.10?

**Answer**: Different scales

```python
# M-values: log2(β / (1-β))
# logFC is difference in M-values, not β-values

# Convert logFC back to Δβ
from methylation_engine.core.planner import delta_beta_to_delta_m
import numpy as np

logFC = 1.5
baseline_beta = 0.5
m_baseline = np.log2(baseline_beta / (1 - baseline_beta))
m_new = m_baseline + logFC
beta_new = 2**m_new / (1 + 2**m_new)
delta_beta = beta_new - baseline_beta  # ≈0.17

# OR use meanB columns
delta_beta_direct = results['meanB_Tumor'] - results['meanB_Normal']
```

### Significant by p-value but small effect

**Question**: padj < 0.05 but |logFC| < 0.5?

**Answer**: Large sample sizes detect small effects

**Solutions**:

```python
# Apply effect size threshold
sig = results[
    (results['padj'] < 0.05) &
    (abs(results['logFC']) > 1.5)
]

# OR use delta-beta threshold
sig = results[
    (results['padj'] < 0.05) &
    (abs(results['meanB_Tumor'] - results['meanB_Normal']) > 0.10)
]
```

### Different results with different shrinkage

**Question**: Why do results change with shrinkage method?

**Answer**: Different prior assumptions

**Recommendations**:

```python
# For publication: Use 'auto' or 'smyth'
results = fit_differential(M, design, shrink='auto')

# For exploration: Use fixed (10.0) for speed
results = fit_differential(M, design, shrink=10.0)

# Difference is typically <1% in detection rate
```

## Platform-Specific Issues

### 450K array

**Issue**: Platform discontinued, limited reagents

**Solutions**:

- Use existing data for replication
- Switch to EPIC for new studies
- Consider 450K-to-EPIC conversion tools

### EPIC array

**Issue**: Some CpGs fail QC more often

**Solutions**:

```python
# Use detection p-value threshold (if available)
M = M[detection_pval < 0.01]

# Filter by call rate
call_rate = (~M.isna()).mean(axis=1)
M = M[call_rate > 0.95]
```

### WGBS

**Issue**: Uneven coverage across CpGs

**Solutions**:

```python
# Filter by minimum coverage
if 'coverage' in M.columns:
    M = M[M['coverage'] >= 10]

# Weight by coverage in analysis
# (Not currently supported - use external tools)
```

## Common Error Messages

### “ValueError: Contrast length != design columns”

**Fix**: Ensure contrast vector length matches design

```python
print(f"Design columns: {design.shape[1]}")
print(f"Contrast length: {len(contrast)}")
contrast = np.array([0, 1, 0])  # Match exactly
```

### “LinAlgError: Singular matrix”

**Fix**: Design matrix is singular (redundant covariates)

```python
# Check for linear dependencies
import numpy as np
rank = np.linalg.matrix_rank(design.values)
if rank < design.shape[1]:
    # Remove redundant columns
    # E.g., don't include both [0,1] and [1,0] for same factor
```

### “KeyError: Sample names”

**Fix**: Index mismatch between M and design

```python
# Align indices
common_samples = M.columns.intersection(design.index)
M = M[common_samples]
design = design.loc[common_samples]
```

### “All chunks failed to process”

**Fix**: Issue in chunking logic

```python
# Try smaller chunk size
results = fit_differential_chunked(M, design, chunk_size=1000)

# OR use standard method on subset
M_subset = M.iloc[:10000]
results = fit_differential(M_subset, design, ...)
```

## Getting Help

### Before opening an issue:

1. **Check this troubleshooting guide**
2. **Review demo scripts**: `demos/planner_demo.py`, `demos/engine_demo.py`
3. **Search existing issues**: [GitHub Issues](https://github.com/dare-afolabi/methylation-engine/issues)

### When opening an issue:

Follow issue templates and include:

```python
# 1. Version info
import core
print(core.__version__)

# 2. Data shapes
print(f"M shape: {M.shape}")
print(f"Design shape: {design.shape}")

# 3. Minimal reproducible example
import numpy as np
import pandas as pd
np.random.seed(42)
M = pd.DataFrame(np.random.randn(1000, 20))
design = pd.DataFrame({'Intercept': 1, 'Group': [0]*10 + [1]*10})
# ... code that produces error

# 4. Full error traceback
# Include complete error message

# 5. Expected vs actual behavior
```

### Contact

- **Issues**: [GitHub Issues](https://github.com/dare-afolabi/methylation-engine/issues)
- **Discussions**: [GitHub Discussions](https://github.com/dare-afolabi/methylation-engine/discussions/5)
- **Email**: [dare.afolabi@outlook.com](mailto:dare.afolabi@outlook.com)