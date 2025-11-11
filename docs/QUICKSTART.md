# Quick Start Guide

## 5-Minute Tutorial

### Installation

```bash
git clone https://github.com/dare-afolabi/methylation-engine.git
cd methylation-engine
pip install -r requirements.txt
```

### Run Demo Analysis

```bash
python demo.py
```

This generates a full analysis report in `log/TIMESTAMP/report.pdf`

-----

## Your First Analysis

### Step 1: Prepare Your Data

**Required files:**

- `M_values.csv` - CpGs × samples, M-values
- `metadata.csv` - Sample information

```python
import pandas as pd

# Load M-values
M = pd.read_csv("M_values.csv", index_col=0)
# Rows = CpGs (e.g., cg00000029)
# Columns = Samples (e.g., Sample_1, Sample_2)

# Load metadata
metadata = pd.read_csv("metadata.csv", index_col=0)
# Must have columns: Type (Normal/Tumor), Batch (optional)
```

### Step 2: Create Analysis Script

```python
#!/usr/bin/env python
import numpy as np
import pandas as pd
from core.engine import (
    filter_cpgs_by_missingness,
    impute_missing_values_fast,
    fit_differential,
    get_significant_cpgs,
    plot_volcano_enhanced
)

# 1. Load data
M = pd.read_csv("M_values.csv", index_col=0)
metadata = pd.read_csv("metadata.csv", index_col=0)

# 2. Quality control
M_filtered, _, _ = filter_cpgs_by_missingness(
    M, max_missing_rate=0.2,
    min_samples_per_group=5,
    groups=metadata['Type']
)

M_imputed = impute_missing_values_fast(M_filtered, method='knn', k=5)

# 3. Create design matrix
design = pd.DataFrame({
    "Intercept": 1,
    "Tumor": (metadata['Type'] == 'Tumor').astype(int)
}, index=M.columns)

# 4. Run differential analysis
results = fit_differential(
    M_imputed,
    design,
    contrast=np.array([0, 1]),  # Test Tumor vs Normal
    shrink='auto',
    robust=True
)

# 5. Extract significant CpGs
sig_cpgs = get_significant_cpgs(
    results,
    lfc_thresh=1.5,
    pval_thresh=0.05,
    direction=None  # Both hyper and hypo
)

print(f"Found {len(sig_cpgs)} significant CpGs")

# 6. Visualize
plot_volcano_enhanced(
    results,
    lfc_thresh=1.5,
    pval_thresh=0.05,
    save_path="volcano.png"
)

# 7. Export results
results.to_csv("differential_results.csv")
```

### Step 3: Run Analysis

```bash
python my_analysis.py
```

**Output:**

- `differential_results.csv` - Full results table
- `volcano.png` - Volcano plot

-----

## Common Use Cases

### 1. Two-Group Comparison (Cancer vs Normal)

```python
design = pd.DataFrame({
    "Intercept": 1,
    "Cancer": [0, 0, 0, 1, 1, 1]  # 0=Normal, 1=Cancer
}, index=M.columns)

results = fit_differential(
    M, design,
    contrast=np.array([0, 1]),  # Test cancer effect
    shrink='auto'
)
```

### 2. Batch Effect Adjustment

```python
# Include batch as covariate
design = pd.DataFrame({
    "Intercept": 1,
    "Cancer": [0, 0, 0, 1, 1, 1],
    "Batch": [0, 1, 0, 1, 0, 1]
}, index=M.columns)

# Test cancer effect (adjusted for batch)
results = fit_differential(
    M, design,
    contrast=np.array([0, 1, 0]),  # Test only cancer
    shrink='auto'
)
```

### 3. Large Arrays (450K/EPIC)

```python
from core.engine import fit_differential_chunked

# Process in chunks to save memory
results = fit_differential_chunked(
    M_large,
    design,
    chunk_size=10000,  # 10K CpGs per chunk
    contrast=np.array([0, 1]),
    shrink=10.0,  # Fixed shrinkage for speed
    verbose=True
)
```

### 4. Extract Hypermethylated CpGs Only

```python
hyper_cpgs = get_significant_cpgs(
    results,
    lfc_thresh=1.5,
    pval_thresh=0.05,
    direction='hyper'  # Only hypermethylated
)
```

### 5. Filter by Delta-Beta

```python
# Require at least 10% methylation difference
sig_cpgs = get_significant_cpgs(
    results,
    lfc_thresh=1.0,
    pval_thresh=0.05,
    delta_beta_thresh=0.10
)
```

-----

## Understanding Output

### Results Columns

|Column          |Description           |Interpretation           |
|----------------|----------------------|-------------------------|
|**logFC**       |Log2 fold change      |Effect size on M-scale   |
|**t**           |Moderated t-statistic |Test statistic           |
|**pval**        |Raw p-value           |Before multiple testing  |
|**padj**        |Adjusted p-value      |FDR-corrected (use this!)|
|**meanM_Normal**|Mean M-value in Normal|Baseline methylation     |
|**meanM_Tumor** |Mean M-value in Tumor |Methylation in condition |
|**meanB_Normal**|Mean beta (Normal)    |% methylation (0-1)      |
|**meanB_Tumor** |Mean beta (Tumor)     |% methylation (0-1)      |

### Interpreting logFC

|logFC|Delta-Beta (approx)|Interpretation   |
|-----|-------------------|-----------------|
|1.0  |0.08-0.15          |Small effect     |
|1.5  |0.12-0.20          |Moderate effect  |
|2.0  |0.15-0.30          |Large effect     |
|2.5+ |0.20-0.40          |Very large effect|

**Note**: Conversion from logFC to Δβ depends on baseline methylation level.

### Significance Thresholds

**Standard thresholds:**

- `padj < 0.05` - 5% FDR (5 false discoveries per 100 calls)
- `|logFC| > 1.5` - Moderate effect size

**Stringent thresholds:**

- `padj < 0.01` - 1% FDR
- `|logFC| > 2.0` - Large effect size

-----

## Troubleshooting

### “Residual degrees of freedom <= 0”

**Problem**: Too many covariates for your sample size

**Solution**:

```python
# Check your design
print(f"Samples: {design.shape[0]}")
print(f"Covariates: {design.shape[1]}")
# Need: samples > covariates + 1
```

### High memory usage

**Problem**: Dataset too large for RAM

**Solution**:

```python
# Use chunked processing
results = fit_differential_chunked(
    M, design,
    chunk_size=5000,  # Reduce chunk size
    ...
)
```

### No significant CpGs found

**Problem**: Insufficient power or no real differences

**Check**:

```python
# 1. Check effect sizes in top hits
print(results[['logFC', 'pval']].head(20))

# 2. Try less stringent threshold
sig = results[results['padj'] < 0.10]  # 10% FDR

# 3. Check if batch effects are masking signal
plot_sample_qc(M, metadata, group_col='Type')
```

### P-value inflation

**Problem**: Q-Q plot shows deviation from diagonal

**Check**:

```python
from core.engine import plot_pvalue_qq
plot_pvalue_qq(results)

# Likely causes:
# 1. Unmodeled batch effects → Add batch to design
# 2. Outlier samples → Check sample QC
# 3. Data not normalized → Check M-value distribution
```

-----

## Next Steps

1. **Read the full README** for advanced usage
1. **Run the demo** to see all features
1. **Check the fact sheet** for technical details
1. **Join discussions** on GitHub for questions

-----

## Getting Help

- **Issues**: [https://github.com/dare-afolabi/methylation-engine/issues](https://github.com/dare-afolabi/methylation-engine/issues)
- **Discussions**: [https://github.com/discussions](https://github.com/discussions)
- **Email**: [dare.afolabi@outlook.com](mailto:dare.afolabi@outlook.com)

## Citation

```bibtex
@software{methylation_engine,
  title = {Differential Methylation Analysis Pipeline},
  year = {2025},
  url = {https://github.com/dare-afolabi/methylation-engine}
}
```