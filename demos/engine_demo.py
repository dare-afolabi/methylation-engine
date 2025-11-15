#!/usr/bin/env python
# coding: utf-8

"""
Differential Methylation Analysis Demo
Optimized for large datasets
"""

import os
import time
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import stats

# Assuming discovery.py contains all functions
from methylation_engine.core.engine import (
    filter_cpgs_by_missingness,
    impute_missing_values_fast,
    filter_min_per_group,
    fit_differential,
    fit_differential_chunked,
    get_significant_cpgs,
    summarize_differential_results,
    plot_sample_qc,
    plot_mean_variance,
    plot_pvalue_qq,
    plot_residual_diagnostics,
    plot_volcano_enhanced,
    export_results,
    PDFLogger,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "random_seed": 1500,
    "n_cpg": 100000,  # Large dataset for chunking demo
    "n_normal": 12,
    "n_tumor": 12,
    "effect_size": 1.5,  # M value +/- effect_size
    "prop_dm": 0.05,
    "missing_rate": 0.05,
    "batch_effect": 0.5,
    "pval_threshold": 0.05,
    "lfc_threshold": 1.0,
    "shrink_method": 10.0,  # Fixed (10.0) is faster than 'auto'
    "max_d0": 50.0,
    "imputation_method": "median",  # Orders faster than KNN for large n CpGs
    # "imputation_k": 5,
    "chunk_size": 50000,  # Process 50K CpGs at a time
    "use_chunked": True,  # Set to False for < 50K CpGs
}

# Configure logger
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs(f"log/{timestamp}/assets", exist_ok=True)
report_path = f"log/{timestamp}"
pdf = PDFLogger(f"{report_path}/report.pdf", echo=True)

pdf.log_text("# Differential Methylation Analysis Report")
pdf.log_text(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Log configuration
pdf.log_text("\n## Analysis Configuration")
for key, value in CONFIG.items():
    pdf.log_text(f"- **{key}**: {value}")

# Speed optimization note
if CONFIG["shrink_method"] == 10:
    pdf.log_text(
        "\n**Performance Note**: Using fixed shrinkage (d0=10) for optimal speed"
    )
    pdf.log_text("- Faster than auto/Smyth shrinkage")
    pdf.log_text("- Minimal impact on results (<1% difference in practice)")

# ============================================================================
# SECTION 1: DATA SIMULATION & MISSING DATA HANDLING
# ============================================================================

print("=" * 70)
pdf.log_text("\n## 1. Data Simulation & Quality Control")
print("=" * 70)

start_time = time.time()

# Simulate data
np.random.seed(CONFIG["random_seed"])
n_cpg = CONFIG["n_cpg"]
n_normal = CONFIG["n_normal"]
n_tumor = CONFIG["n_tumor"]
effect_size = CONFIG["effect_size"]
prop_dm = CONFIG["prop_dm"]

samples = [f"Normal_{i}" for i in range(n_normal)] + [
    f"Tumor_{i}" for i in range(n_tumor)
]
metadata = pd.DataFrame(
    {"Sample": samples, "Type": ["Normal"] * n_normal + ["Tumor"] * n_tumor}
).set_index("Sample")

# Generate M-values
a_trunc, b_trunc = (-1.2 - 0) / 2.0, (0.8 - 0) / 2.0
M = stats.truncnorm.rvs(
    a_trunc, b_trunc, loc=0.2, scale=2.0, size=(n_cpg, n_normal + n_tumor)
)

# Add differential methylation
n_dm = int(n_cpg * prop_dm)
dm_indices = np.random.choice(n_cpg, n_dm, replace=False)
top = round(len(dm_indices) / 1.25)
hyper_indices = dm_indices[:top]
hypo_indices = dm_indices[top:]

M[hyper_indices, n_normal:] += effect_size
M[hypo_indices, n_normal:] -= effect_size

M_df = pd.DataFrame(M, index=[f"cg{i:08d}" for i in range(n_cpg)], columns=samples)

pdf.log_text("\n### Data Simulation")
pdf.log_text(f"- **Total CpGs**: {n_cpg:,}")
pdf.log_text(
    f"- **Samples**: {n_normal + n_tumor} ({n_normal} Normal, {n_tumor} Tumor)"
)
pdf.log_text(f"- **True DM CpGs**: {n_dm:,} ({prop_dm*100:.1f}%)")
pdf.log_text(f"- **Hypermethylated**: {len(hyper_indices):,}")
pdf.log_text(f"- **Hypomethylated**: {len(hypo_indices):,}")
pdf.log_text(f"- **Effect size**: ±{effect_size}")

# Introduce missing values
missing_mask = np.random.random(M_df.shape) < CONFIG["missing_rate"]
M_df_missing = M_df.copy()
M_df_missing[missing_mask] = np.nan

pdf.log_text("\n### Missing Data")
pdf.log_text(
    f"- **Missing values**: {M_df_missing.isna().sum().sum():,} ({M_df_missing.isna().sum().sum() / M_df_missing.size * 100:.2f}%)"
)

# Filter by missingness
M_df_filtered, filtered, retained = filter_cpgs_by_missingness(
    M_df_missing,
    max_missing_rate=0.2,
    min_samples_per_group=5,
    groups=metadata["Type"],
)

pdf.log_text(f"- **Filtered CpGs**: {filtered:,}")
pdf.log_text(f"- **Retained CpGs**: {retained:,}")

# Impute remaining missing values
M_df_imputed = impute_missing_values_fast(
    M_df_filtered,
    method=CONFIG["imputation_method"],
    # k=CONFIG["imputation_k"],
)

pdf.log_text(
    f"- **After imputation**: {M_df_imputed.isna().sum().sum()} missing values"
)

if M_df_imputed.isna().mean().mean() > 0.15:
    pdf.log_text(
        "- **!** High residual missingness (>15%); consider stricter filtering."
    )

# Filter per-group counts
M_ready = filter_min_per_group(
    M_df_imputed,
    groups=metadata["Type"],
    min_per_group=5,
    verbose=True,
)

pdf.log_text(f"- **After filtering by group counts**: {M_ready.shape[0]:,} CpGs remain")

elapsed = time.time() - start_time
pdf.log_text(f"\n✔ Completed in {elapsed:.2f} seconds")

# ============================================================================
# SECTION 2: SAMPLE QC VISUALIZATION
# ============================================================================

print("\n" + "=" * 70)
pdf.log_text("\n## 2. Sample-Level Quality Control")
print("=" * 70)

start_time_2 = time.time()

fig = plot_sample_qc(
    M_df_missing,
    metadata,
    group_col="Type",
    save_path=f"{report_path}/assets/figure_1_sample_qc.png",
)
pdf.log_image(
    f"{report_path}/assets/figure_1_sample_qc.png",
    "Figure 1",
    "Sample QC metrics by group",
)

elapsed = time.time() - start_time_2
pdf.log_text(f"\n✔ Completed in {elapsed:.2f} seconds")

# ============================================================================
# SECTION 3: DIFFERENTIAL ANALYSIS (CHUNKED OR STANDARD)
# ============================================================================

print("\n" + "=" * 70)
pdf.log_text("\n## 3. Differential Methylation Analysis")
print("=" * 70)

start_time_3 = time.time()

# Design matrix
sample_type = (metadata["Type"] == "Tumor").astype(float)
design = pd.DataFrame(
    {"Intercept": 1.0, "Tumor": sample_type.values}, index=metadata.index
)

contrast = np.array([0.0, 1.0])

# Choose method based on dataset size
if CONFIG["use_chunked"] and len(M_ready) >= 50000:
    pdf.log_text(f"\n### Using Chunked Analysis ({len(M_ready):,} CpGs)")
    pdf.log_text(f"- **Chunk size**: {CONFIG['chunk_size']:,} CpGs")
    pdf.log_text(
        f"- **Expected chunks**: {int(np.ceil(len(M_ready) / CONFIG['chunk_size']))}"
    )

    res = fit_differential_chunked(
        M_ready,
        design,
        chunk_size=CONFIG["chunk_size"],
        contrast=contrast,
        shrink=CONFIG["shrink_method"],
        robust=True,
        max_d0=CONFIG["max_d0"],
        verbose=True,  # Shows progress
    )
else:
    pdf.log_text(f"\n### Using Standard Analysis ({len(M_ready):,} CpGs)")
    pdf.log_text("- Dataset small enough for standard method")

    res = fit_differential(
        M_ready,
        design,
        contrast=contrast,
        shrink=CONFIG["shrink_method"],
        robust=True,
        return_residuals=False,
        max_d0=CONFIG["max_d0"],
    )

pdf.log_text("\n### Top 5 Significant CpGs")
pdf.log_dataframe(
    res[["logFC", "se", "t", "pval", "padj", "meanM_Normal", "meanM_Tumor"]].head()
)

# Summary statistics
summary = summarize_differential_results(res, pval_thresh=CONFIG["pval_threshold"])

pdf.log_text("\n### Summary Statistics")
pdf.log_text(f"- **Total CpGs tested**: {summary['total_tested']:,}")
pdf.log_text(
    f"- **Significant CpGs**: {summary['significant']:,} ({summary['pct_significant']:.1f}%)"
)
pdf.log_text(f"- **Hypermethylated**: {summary['hypermethylated']:,}")
pdf.log_text(f"- **Hypomethylated**: {summary['hypomethylated']:,}")
pdf.log_text(f"- **Mean |logFC| (sig)**: {summary['mean_logFC_sig']:.2f}")
pdf.log_text(f"- **Median |logFC| (sig)**: {summary['median_logFC_sig']:.2f}")
pdf.log_text(f"- **Max |logFC|**: {summary['max_abs_logFC']:.2f}")
pdf.log_text(f"- **Min p-value**: {summary['min_pval']:.2e}")

pdf.log_text("\n### Variance Shrinkage")
pdf.log_text(f"- **Prior df (d0)**: {summary['d0']:.1f}")
pdf.log_text(f"- **Shrinkage factor**: {summary['shrinkage_factor']:.2f}x")
pdf.log_text(f"- **Median raw variance**: {res['s2'].median():.4f}")
pdf.log_text(f"- **Median moderated variance**: {res['s2_post'].median():.4f}")

# Performance metrics
true_positives = len(
    set(res[res["padj"] < 0.05].index) & set([f"cg{i:08d}" for i in dm_indices])
)
sensitivity = true_positives / n_dm * 100 if n_dm > 0 else 0

pdf.log_text("\n### Performance (vs. Known Truth)")
pdf.log_text(f"- **True detection**: {true_positives:,}/{n_dm:,}")
pdf.log_text(f"- **Sensitivity**: {sensitivity:.1f}%")

# Analysis speed
elapsed = time.time() - start_time_3
rate = len(res) / elapsed
pdf.log_text("\n### Performance Metrics")
pdf.log_text(f"- **Analysis time**: {elapsed:.2f}s")
pdf.log_text(f"- **Processing rate**: {rate:.0f} CpGs/sec")

pdf.log_text(f"\n✔ Completed in {elapsed:.2f} seconds")

# ============================================================================
# SECTION 4: DIAGNOSTIC PLOTS (Sample from results)
# ============================================================================

print("\n" + "=" * 70)
pdf.log_text("\n## 4. Diagnostic Plots")
print("=" * 70)

start_time_4 = time.time()

# For large datasets, only generate diagnostics on subset or top hits
if len(res) > 10000:
    pdf.log_text("\n**Note**: Generating diagnostics on subset for speed")
    # Get top 5000 by variance for diagnostics
    top_var = res["s2"].nlargest(5000).index
    M_subset = M_ready.loc[top_var]

    res_diag, residuals = fit_differential(
        M_subset,
        design,
        contrast=contrast,
        shrink=CONFIG["shrink_method"],
        robust=True,
        return_residuals=True,
        max_d0=CONFIG["max_d0"],
    )
else:
    # Refit with residuals for diagnostics
    res_diag, residuals = fit_differential(
        M_ready,
        design,
        contrast=contrast,
        shrink=CONFIG["shrink_method"],
        robust=True,
        return_residuals=True,
        max_d0=CONFIG["max_d0"],
    )

# Mean-variance relationship
fig = plot_mean_variance(
    res_diag, save_path=f"{report_path}/assets/figure_2_mean_variance.png"
)
pdf.log_image(
    f"{report_path}/assets/figure_2_mean_variance.png",
    "Figure 2",
    "Mean-variance relationship and shrinkage effect",
)

# P-value Q-Q plot
fig = plot_pvalue_qq(res_diag, save_path=f"{report_path}/assets/figure_3_pvalue_qq.png")
pdf.log_image(
    f"{report_path}/assets/figure_3_pvalue_qq.png",
    "Figure 3",
    "P-value distribution (inflation check)",
)

# Residual diagnostics
fig = plot_residual_diagnostics(
    residuals,
    M_subset if len(res) > 10000 else M_ready,
    design,
    top_n=9,
    save_path=f"{report_path}/assets/figure_4_residuals.png",
)
pdf.log_image(
    f"{report_path}/assets/figure_4_residuals.png",
    "Figure 4",
    "Residual Q-Q plots for top variable CpGs",
)

elapsed = time.time() - start_time_4
pdf.log_text(f"\n✔ Completed in {elapsed:.2f} seconds")

# ============================================================================
# SECTION 5: FEATURE EXTRACTION
# ============================================================================

print("\n" + "=" * 70)
pdf.log_text(
    f"\n## 5. Feature Extraction (at |logFC| > {CONFIG['lfc_threshold']} cutoff)"
)
print("=" * 70)

start_time_5 = time.time()

# Get significant CpGs with summary
summary_all = get_significant_cpgs(
    res,
    lfc_col="logFC",
    pval_col="padj",
    lfc_thresh=CONFIG["lfc_threshold"],
    pval_thresh=CONFIG["pval_threshold"],
    direction=None,
    return_summary=True,
)

pdf.log_text("\n### Significant CpGs")
pdf.log_text(f"- **Total**: {summary_all['n_significant']:,}")
pdf.log_text(f"- **Hypermethylated**: {summary_all['n_hyper']:,}")
pdf.log_text(f"- **Hypomethylated**: {summary_all['n_hypo']:,}")
pdf.log_text(f"- **Mean |logFC|**: {summary_all['mean_abs_lfc']:.2f}")

# Get hypermethylated only
hyper_cpgs = get_significant_cpgs(
    res,
    lfc_thresh=CONFIG["lfc_threshold"],
    pval_thresh=CONFIG["pval_threshold"],
    direction="hyper",
)

pdf.log_text(f"\n### Hypermethylated CpGs ({len(hyper_cpgs):,})")
if len(hyper_cpgs) > 0:
    # Show top 21 only
    display_cpgs = hyper_cpgs[:21]
    pdf.log_text(f"{', '.join(display_cpgs)}")
    if len(hyper_cpgs) > 21:
        pdf.log_text(f"... {len(hyper_cpgs) - 21:,} more")

# Get hypomethylated only
hypo_cpgs = get_significant_cpgs(
    res,
    lfc_thresh=CONFIG["lfc_threshold"],
    pval_thresh=CONFIG["pval_threshold"],
    direction="hypo",
)

pdf.log_text(f"\n### Hypomethylated CpGs ({len(hypo_cpgs):,})")
if len(hypo_cpgs) > 0:
    display_cpgs = hypo_cpgs[:21]
    pdf.log_text(f"{', '.join(display_cpgs)}")
    if len(hypo_cpgs) > 21:
        pdf.log_text(f"... {len(hypo_cpgs) - 21:,} more")

elapsed = time.time() - start_time_5
pdf.log_text(f"<br></br>✔ Completed in {elapsed:.2f} seconds")

# ============================================================================
# SECTION 6: BATCH-ADJUSTED ANALYSIS
# ============================================================================

print("\n" + "=" * 70)
pdf.log_text("\n## 6. Batch-Adjusted Analysis")
print("=" * 70)

start_time_6 = time.time()

# Add batch effect to data
batch_effect = CONFIG["batch_effect"]
types = ["Normal"] * n_normal + ["Tumor"] * n_tumor
batch_labels = (
    [0] * (n_normal // 2)
    + [1] * (n_normal - n_normal // 2)
    + [0] * (n_tumor // 2)
    + [1] * (n_tumor - n_tumor // 2)
)

sample_names = [f"{t}_{b}_{i}" for i, (t, b) in enumerate(zip(types, batch_labels))]

metadata_batch = pd.DataFrame(
    {"Type": types, "Batch": batch_labels}, index=sample_names
)

# Simulate with batch effect
M_batch = stats.truncnorm.rvs(
    a_trunc, b_trunc, loc=0.2, scale=2.0, size=(n_cpg, n_normal + n_tumor)
)

n_dm_batch = int(n_cpg * prop_dm)
dm_indices_batch = np.random.choice(n_cpg, n_dm_batch, replace=False)
top_batch = round(len(dm_indices_batch) / 1.25)
hyper_indices_batch = dm_indices_batch[:top_batch]
hypo_indices_batch = dm_indices_batch[top_batch:]

M_batch[hyper_indices_batch, n_normal:] += effect_size
M_batch[hypo_indices_batch, n_normal:] -= effect_size

# Apply batch effect
batch_idx = [i for i, b in enumerate(batch_labels) if b == 1]
M_batch[:, batch_idx] -= batch_effect

M_batch_df = pd.DataFrame(
    M_batch, index=[f"cg{i:08d}" for i in range(n_cpg)], columns=sample_names
)

pdf.log_text("\n### Batch Effect Design")
pdf.log_text(f"- **Batch effect size**: {batch_effect}")
pdf.log_text("- **Batch distribution**: 50/50 in each group")

try:
    # Design matrix with batch
    design_batch = pd.DataFrame({"const": 1}, index=M_batch_df.columns)
    design_batch["Tumor"] = (metadata_batch["Type"] == "Tumor").astype(int)
    design_batch["Batch"] = metadata_batch["Batch"].astype(int)

    # Validate design
    if design_batch.shape[0] != M_batch_df.shape[1]:
        raise ValueError(
            f"Design rows ({design_batch.shape[0]}) != data columns ({M_batch_df.shape[1]})"
        )

    pdf.log_text("\n### Approach 1: Single Contrast (Recommended)")
    pdf.log_text("**- Tests tumor effect ADJUSTED for batch**")

    # Test ONLY tumor effect (batch is adjusted for automatically)
    contrast_tumor = np.array([0, 1, 0])

    # Use chunked or standard based on size
    if CONFIG["use_chunked"] and len(M_batch_df) >= 50000:
        pdf.log_text(f"\n**Using chunked analysis** ({len(M_batch_df):,} CpGs)")

        res_tumor_adj = fit_differential_chunked(
            M=M_batch_df,
            design=design_batch,
            chunk_size=CONFIG["chunk_size"],
            contrast=contrast_tumor,
            shrink=CONFIG["shrink_method"],  # Use fixed shrinkage for speed
            robust=True,
            max_d0=CONFIG["max_d0"],
            verbose=True,
        )
    else:
        res_tumor_adj = fit_differential(
            M=M_batch_df,
            design=design_batch,
            contrast=contrast_tumor,
            shrink=CONFIG["shrink_method"],
            robust=True,
            max_d0=CONFIG["max_d0"],
        )

    pdf.log_text("\n**Top 5 CpGs (batch-adjusted)**")
    pdf.log_dataframe(res_tumor_adj[["logFC", "t", "pval", "padj"]].head())

    sig_batch_adj = get_significant_cpgs(
        res_tumor_adj,
        pval_thresh=CONFIG["pval_threshold"],
        return_summary=True,
    )

    pdf.log_text("\n**Results Summary**")
    pdf.log_text(f"- **Significant CpGs**: {sig_batch_adj['n_significant']:,}")
    pdf.log_text(f"- **Hypermethylated**: {sig_batch_adj['n_hyper']:,}")
    pdf.log_text(f"- **Hypomethylated**: {sig_batch_adj['n_hypo']:,}")

    # Performance
    true_positives_batch = len(
        set(res_tumor_adj[res_tumor_adj["padj"] < 0.05].index)
        & set([f"cg{i:08d}" for i in dm_indices_batch])
    )
    sensitivity_batch = true_positives_batch / n_dm_batch * 100 if n_dm_batch > 0 else 0

    pdf.log_text(f"- **True detections**: {true_positives_batch:,}/{n_dm_batch:,}")
    pdf.log_text(f"- **Sensitivity**: {sensitivity_batch:.1f}%")

    pdf.log_text("\n✔ Batch-adjusted analysis completed successfully")

    # F-test for comparison (optional - usually skip for large datasets)
    if len(M_batch_df) < 50000:  # Only for smaller datasets
        pdf.log_text("\n### Approach 2: Multi-Coefficient F-test")
        pdf.log_text("- **Tests if tumor OR batch (or both) have effects**")
        pdf.log_text("- **!** Use for QC, not for effect size interpretation")

        R = np.array([[0, 1, 0], [0, 0, 1]])  # Tumor effect  # Batch effect

        resF = fit_differential(
            M=M_batch_df,
            design=design_batch,
            contrast_matrix=R,
            shrink=CONFIG["shrink_method"],
            robust=True,
            max_d0=CONFIG["max_d0"],
        )

        # Extract coefficients (for interpretation only, not effect sizes)
        beta_hat = np.linalg.pinv(design_batch.values) @ M_batch_df.T.values
        resF["logFC_Tumor_partial"] = (R[0] @ beta_hat).flatten()
        resF["logFC_Batch_partial"] = (R[1] @ beta_hat).flatten()

        pdf.log_text("\n**Top 5 CpGs by F-test**")
        pdf.log_dataframe(
            resF[
                [
                    "F",
                    "pval",
                    "padj",
                    "logFC_Tumor_partial",
                    "logFC_Batch_partial",
                ]
            ].head()
        )

        pdf.log_text(
            "\n- **!** **Note**: Partial coefficients shown above are NOT marginal effects."
        )
        pdf.log_text("- Use Approach 1 (single contrast) for tumor effect sizes.")
    else:
        pdf.log_text("\n### ! Approach 2: F-test skipped")
        pdf.log_text("- Dataset too large for demonstration F-test")
        pdf.log_text("- Use single contrast (Approach 1) for production analyses")

except Exception as e:
    pdf.log_text(f"\n**!** Error in batch-adjusted analysis: {str(e)}")
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()

elapsed = time.time() - start_time_6
pdf.log_text(f"\n✔ Completed in {elapsed:.2f} seconds")

# ============================================================================
# SECTION 7: VISUALIZATION
# ============================================================================

print("\n" + "=" * 70)
pdf.log_text("\n## 7. Visualization")
print("=" * 70)

start_time_7 = time.time()

if res_tumor_adj is None:
    pdf.log_text("Batch-adjusted result missing; using previous res for plotting")
    res_for_plot = res
else:
    res_for_plot = res_tumor_adj

fig = plot_volcano_enhanced(
    res_for_plot,
    lfc_thresh=CONFIG["lfc_threshold"],
    pval_thresh=CONFIG["pval_threshold"],
    top_n=10,
    save_path=f"{report_path}/assets/figure_5_volcano.png",
)
pdf.log_image(
    f"{report_path}/assets/figure_5_volcano.png",
    "Figure 5",
    "Volcano plot with top 10 hits labeled",
)

elapsed = time.time() - start_time_7
pdf.log_text(f"\n✔ Completed in {elapsed:.2f} seconds")

# ============================================================================
# SECTION 8: EXPORT RESULTS
# ============================================================================

print("\n" + "=" * 70)
pdf.log_text("\n## 8. Export Results")
print("=" * 70)

start_time_7 = time.time()

# Export significant CpGs
sig_results = res_tumor_adj[res_tumor_adj["padj"] < CONFIG["pval_threshold"]]
if len(sig_results) > 0:
    export_results(
        sig_results,
        output_path=f"{report_path}/significant_cpgs.csv",
        format="csv",
        include_all=False,
    )
    pdf.log_text(
        f"- **Significant CpGs exported**: {len(sig_results):,} to `significant_cpgs.csv`"
    )
else:
    pdf.log_text("- **No significant CpGs** to export")

# Export all results (if reasonable size)
if len(res) < 500000:
    export_results(
        res_tumor_adj,
        output_path=f"{report_path}/all_results.csv",
        format="csv",
        include_all=True,
    )
    pdf.log_text(f"- **All results exported**: {len(res):,} to `all_results.csv`")
else:
    pdf.log_text(
        f"- **All results**: Skipped export ({len(res):,} CpGs would create {len(res)*len(res.columns)*10/1e6:.1f}MB file)"
    )

elapsed = time.time() - start_time_7
pdf.log_text(f"\n✔ Completed in {elapsed:.2f} seconds")


# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 70)
pdf.log_text("\n## Summary")
print("=" * 70)

total_time = time.time() - start_time

pdf.log_text("\n### Analysis Complete!")
pdf.log_text(f"- **Total runtime**: {total_time:.1f}s ({total_time/60:.1f} minutes)")
pdf.log_text(f"- **Results directory**: `{report_path}/`")
pdf.log_text(f"- **CpGs analyzed**: {len(res):,}")
pdf.log_text(f"- **Significant CpGs**: {summary['significant']:,}")
pdf.log_text(f"- **Sensitivity**: {sensitivity:.1f}%")
pdf.log_text(f"- **Significant CpGs with batch adjustment**: {sig_batch_adj['n_significant']:,}")
pdf.log_text(f"- **Sensitivity with batch adjustment**: {sensitivity_batch:.1f}%")

# Performance summary
pdf.log_text("\n### Performance Summary")
pdf.log_text(f"- **Processing rate**: {len(res)/total_time:.0f} CpGs/sec")
method_is_chunked = CONFIG.get("use_chunked", False)
pdf.log_text(f"- **Method**: {'Chunked' if method_is_chunked else 'Standard'}")
if method_is_chunked:
    pdf.log_text(f"- **Chunk size**: {CONFIG['chunk_size']:,} CpGs")
pdf.log_text(
    f"- **Shrinkage**: {'Fixed (d0=' + str(CONFIG['shrink_method']) + ')' if isinstance(CONFIG['shrink_method'], (int, float)) else CONFIG['shrink_method']}"
)

# Extrapolation to standard arrays
if total_time > 0 and len(res) > 0:
    final_rate = len(res) / total_time
    pdf.log_text("\n### Estimated Processing Times")
    pdf.log_text("Based on current processing rate:")
    pdf.log_text(
        f"- **450K array** (~450,000 CpGs): {450000/final_rate/60:.1f} minutes"
    )
    pdf.log_text(
        f"- **EPIC array** (~850,000 CpGs): {850000/final_rate/60:.1f} minutes"
    )
    pdf.log_text(f"- **EPIC v2** (~935,000 CpGs): {935000/final_rate/60:.1f} minutes")

# Performance tips
pdf.log_text("\n### Performance Tips")
if total_time > 300:  # > 5 minutes
    pdf.log_text("- **Speed**: Consider increasing chunk_size to 100,000")
    pdf.log_text("- **Speed**: Use fixed shrinkage (d0=10) instead of 'auto'")
    pdf.log_text("- **Memory**: Decrease chunk_size if encountering memory errors")
    pdf.log_text("- **Filtering**: Remove low-variance CpGs before analysis")
elif total_time < 60:
    pdf.log_text("✔ Analysis completed very efficiently!")
else:
    pdf.log_text("✔ Good performance - no optimization needed")

# Data quality notes
pdf.log_text("\n### Data Quality Notes")
if sensitivity < 50:
    pdf.log_text(f"**!** **Low sensitivity ({sensitivity:.1f}%)**: Consider:")
    pdf.log_text("  - Increasing sample size")
    pdf.log_text("  - Checking for batch effects (see Section 6)")
    pdf.log_text("  - Verifying normalization quality")
elif sensitivity < 70:
    pdf.log_text(
        f"**!** **Moderate sensitivity ({sensitivity:.1f}%)**: Acceptable for this effect size"
    )
else:
    pdf.log_text(f"✔ **Good sensitivity ({sensitivity:.1f}%)**")

if summary["pct_significant"] > 20:
    pdf.log_text(
        f"**!** **High % significant ({summary['pct_significant']:.1f}%)**: Check for:"
    )
    pdf.log_text("  - P-value inflation (see Figure 3)")
    pdf.log_text("  - Global methylation differences")
    pdf.log_text("  - Technical artifacts")
elif summary["pct_significant"] < 0.1:
    pdf.log_text(
        f"**! Very low % significant ({summary['pct_significant']:.2f}%)**: Consider:"
    )
    pdf.log_text("  - Lowering logFC threshold")
    pdf.log_text("  - Checking sample size/power")
    pdf.log_text("  - Verifying biological differences exist")
else:
    pdf.log_text(f"✔ **Reasonable % significant**: {summary['pct_significant']:.1f}%")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE!")
print("=" * 70)
print(f"Total runtime: {total_time:.1f}s ({total_time/60:.1f} minutes)")
print(f"Results saved to: {report_path}/")
print(f"PDF report: {report_path}/report.pdf")

# Save PDF
pdf.save()

print("\n" + "=" * 70)
print("FILES GENERATED:")
print("=" * 70)
print(f"1. {report_path}/report.pdf - Full analysis report")
print(f"2. {report_path}/significant_cpgs.csv - Significant CpGs only")
if len(res) < 500000:
    print(f"3. {report_path}/all_results.csv - All tested CpGs")
print(f"4. {report_path}/assets/ - Diagnostic plots (PNG)")
print("=" * 70)
