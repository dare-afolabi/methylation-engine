#!/usr/bin/env python
# coding: utf-8

"""
Differential Methylation Analysis Engine
Production version with bug fixes and enhancements
"""

import os
import re
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    Image,
    ListFlowable,
    ListItem,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
)
from scipy import linalg, optimize, stats
from scipy.special import digamma, polygamma
from sklearn.impute import KNNImputer
from statsmodels.stats.multitest import multipletests

np.random.seed(1500)


# ============================================================================
# INPUT VALIDATION
# ============================================================================


def validate_design(design: pd.DataFrame, M: pd.DataFrame) -> None:
    """Validate design matrix against data."""
    if not isinstance(design, pd.DataFrame):
        raise TypeError("design must be a pandas DataFrame")
    
    if design.shape[1] >= design.shape[0]:
        raise ValueError(
            f"Too many covariates ({design.shape[1]}) for sample\
             size ({design.shape[0]})"
        )

    if design.shape[0] != M.shape[1]:
        raise ValueError(
            f"design rows ({design.shape[0]}) != data columns ({M.shape[1]})"
        )

    if not np.array_equal(design.index.values, M.columns.values):
        raise ValueError("design index must exactly match M columns")

    if design.isnull().any().any():
        raise ValueError("design contains missing values")

    if (design.dtypes == object).any():
        non_numeric = design.select_dtypes(include=[object]).columns.tolist()
        raise ValueError(f"design contains non-numeric columns: {non_numeric}")



def validate_contrast(contrast: np.ndarray, design: pd.DataFrame) -> None:
    """Validate contrast vector against design matrix."""
    contrast = np.asarray(contrast).reshape(-1)

    if contrast.shape[0] != design.shape[1]:
        raise ValueError(
            f"Contrast length ({contrast.shape[0]}) !=\
             design columns ({design.shape[1]})"
        )

    if not np.isfinite(contrast).all():
        raise ValueError("Contrast contains non-finite values")


# ============================================================================
# CORE STATISTICAL FUNCTIONS
# ============================================================================


def _winsorize_array(
    x: np.ndarray, lower_q: float = 0.05, upper_q: float = 0.95
) -> np.ndarray:
    """Clip array values to specified quantiles to reduce outlier influence."""
    lo = np.nanquantile(x, lower_q)
    hi = np.nanquantile(x, upper_q)
    return np.clip(x, lo, hi)


def _estimate_smyth_prior(
    s2: np.ndarray, df_resid: float, robust: bool = True, max_d0: float = 50.0
) -> Tuple[float, float]:
    """
    Estimate empirical Bayes prior (d0, s0²) using Smyth's method.

    Parameters
    ----------
    s2 : np.ndarray
        Raw variance estimates
    df_resid : float
        Residual degrees of freedom
    robust : bool
        Use winsorization for target variance
    max_d0 : float
        Maximum prior df (prevents over-shrinkage in small samples)

    Returns
    -------
    Tuple[float, float]
        (d0, s0_squared) - prior degrees of freedom and prior variance
    """
    s2 = np.asarray(s2, dtype=float)
    s2 = s2[np.isfinite(s2)]
    if s2.size == 0:
        raise ValueError("No finite variances provided")

    if (s2 <= 0).any() or not np.isfinite(s2).all():
        warnings.warn("Invalid variances detected in Smyth prior estimation.")
        return float(min(10.0, df_resid)), float(np.median(s2[s2 > 0]))

    if robust:
        s2_for_target = _winsorize_array(s2, 0.05, 0.95)
    else:
        s2_for_target = s2

    log_s2 = np.log(s2_for_target)
    m = np.mean(log_s2)
    v = np.var(log_s2, ddof=1)

    # Check if shrinkage is beneficial
    min_var_heterogeneity = 0.01

    if v < min_var_heterogeneity:
        warnings.warn(
            f"Variance heterogeneity is low (Var[log(s²)] = {v:.4f}). "
            f"Using conservative shrinkage (d0 = {min(10.0, df_resid):.1f})."
        )
        d0_est = float(min(10.0, df_resid))
        s0_sq = float(np.median(s2_for_target))
        return d0_est, s0_sq

    def f(d0):
        d0 = np.maximum(d0, 1e-8)
        val = polygamma(1, df_resid / 2.0) - polygamma(1, (df_resid + d0) / 2.0)
        return val - v

    try:
        low, high = 1e-8, max_d0
        f_low = f(low)
        f_high = f(high)
        if f_low * f_high < 0:
            d0_est = optimize.brentq(f, low, high, maxiter=200)
        else:
            sol = optimize.minimize_scalar(
                lambda x: (f(x)) ** 2, bounds=(0.0, max_d0), method="bounded"
            )
            d0_est = np.maximum(0.0, sol.x)
    except Exception as e:
        warnings.warn(f"Prior estimation failed: {e}. Using conservative d0.")
        d0_est = float(min(10.0, df_resid))

    d0_est = float(min(d0_est, max_d0))
    log_s0sq = m - (digamma(df_resid / 2.0) - digamma((df_resid + d0_est) / 2.0))
    s0_sq = float(np.exp(log_s0sq))
    s0_sq = np.maximum(s0_sq, 1e-12)

    return d0_est, s0_sq


def _add_group_means(
    res: pd.DataFrame,
    M_df: pd.DataFrame,
    design: pd.DataFrame,
    group_col: Optional[str] = None,
    add_beta: bool = True,
) -> pd.DataFrame:
    """
    Add per-group average M-values and Beta-values to results.

    CRITICAL: M_df must contain all CpGs in res.index
    """
    # Ensure M_df contains all result CpGs
    if not res.index.isin(M_df.index).all():
        missing = set(res.index) - set(M_df.index)
        raise ValueError(
            f"M_df missing {len(missing)} CpGs from results. "
            "This is a bug - contact developers."
        )

    if group_col is None:
        candidates = [
            c
            for c in design.columns
            if design[c].nunique() <= 5 and design[c].nunique() > 1
        ]
        if not candidates:
            raise ValueError("No suitable grouping column found in design.")
        group_col = candidates[0]

    design = design.loc[M_df.columns]
    groups = design[group_col].copy()
    if groups.dtype == float:
        groups = groups.astype(int).astype(str)
    groups = groups.replace({"0": "Normal", "1": "Tumor"})

    # Only use CpGs in results
    M_subset = M_df.loc[res.index]

    meanM = M_subset.T.groupby(groups).mean().T
    for g in meanM.columns:
        res[f"meanM_{g}"] = meanM[g]

    if add_beta:
        beta = 2**M_subset / (1 + 2**M_subset)
        meanB = beta.T.groupby(groups).mean().T
        for g in meanB.columns:
            res[f"meanB_{g}"] = meanB[g]

    return res


# ============================================================================
# MAIN DIFFERENTIAL ANALYSIS
# ============================================================================


def fit_differential(
    M: pd.DataFrame,
    design: pd.DataFrame,
    contrast: Optional[np.ndarray] = None,
    contrast_matrix: Optional[np.ndarray] = None,
    shrink: Union[str, float] = "auto",
    robust: bool = True,
    eps: float = 1e-8,
    return_residuals: bool = False,
    min_count: int = 3,
    max_d0: float = 50.0,
    winsor_lower: float = 0.05,
    winsor_upper: float = 0.95,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Fit linear models with empirical Bayes variance shrinkage.

    Parameters
    ----------
    M : pd.DataFrame
        CpG x samples matrix of M-values (may contain NaN)
    design : pd.DataFrame
        samples x covariates design matrix
    contrast : np.ndarray, optional
        Contrast vector for single coefficient test
    contrast_matrix : np.ndarray, optional
        Matrix for multi-coefficient F-test
    shrink : str or float
        Shrinkage method: 'auto', 'none', 'median', 'smyth', or numeric d0
    robust : bool
        Use robust variance estimation (winsorization)
    eps : float
        Small constant for numerical stability
    return_residuals : bool
        If True, return (results, residuals)
    min_count : int
        Minimum samples required per CpG for fitting
    max_d0 : float
        Maximum prior df for Smyth method
    winsor_lower : float
        Lower quantile for winsorization (if robust=True)
    winsor_upper : float
        Upper quantile for winsorization (if robust=True)

    Returns
    -------
    pd.DataFrame or Tuple
        Results dataframe, optionally with residuals

    Examples
    --------
    >>> M = pd.DataFrame(np.random.randn(1000, 20))
    >>> design = pd.DataFrame({'intercept': 1, 'group': [0]*10 + [1]*10})
    >>> results = fit_differential(M, design, contrast=np.array([0, 1]))
    """
    # Validate inputs
    validate_design(design, M)

    if M.isna().any().any():
        warnings.warn(
            "M contains missing values. Using per-CpG fitting to handle missingness."
        )

    if contrast is not None and contrast_matrix is not None:
        raise ValueError("Provide either contrast or contrast_matrix, not both")

    if contrast is not None:
        validate_contrast(contrast, design)

    Y = M.values
    G, n = Y.shape
    X = design.values
    p = X.shape[1]
    df_resid = n - p

    if df_resid <= 0:
        raise ValueError(
            f"Residual degrees of freedom <= 0 (n={n}, p={p}). "
            "Reduce number of covariates or increase sample size."
        )

    # Initialize storage
    beta_hat_all = np.full((p, G), np.nan)
    s2_all = np.full(G, np.nan)
    n_obs = np.full(G, 0)
    residuals = np.full_like(Y, np.nan)

    # Fit per-CpG models (handles missing data)
    for g in range(G):
        y = Y[g, :]
        mask = ~np.isnan(y)
        n_present = mask.sum()

        if n_present < min_count:
            continue

        n_obs[g] = n_present
        y_obs = y[mask]
        X_obs = X[mask, :]

        if np.var(y_obs) < 1e-12:
            warnings.warn(f"CpG {M.index[g]} has near-zero variance")
            continue

        try:
            XtX = X_obs.T @ X_obs
            XtX_inv = linalg.pinv(XtX)
            beta = XtX_inv @ (X_obs.T @ y_obs)
            beta_hat_all[:, g] = beta

            fitted = X_obs @ beta
            resid = y_obs - fitted
            residuals[g, mask] = resid

            SSE = np.sum(resid**2)
            df_g = n_present - p
            s2_all[g] = SSE / df_g if df_g > 0 else np.nan

        except linalg.LinAlgError:
            warnings.warn(f"Failed to fit CpG {M.index[g]}: singular matrix")
            continue
        except Exception as e:
            warnings.warn(f"Failed to fit CpG {M.index[g]}: {e}")
            continue

    valid = np.isfinite(s2_all)
    if valid.sum() == 0:
        raise ValueError("No CpGs could be fit successfully")

    beta_hat = beta_hat_all[:, valid]
    s2 = s2_all[valid]
    M_valid = M.iloc[valid]
    residuals_valid = residuals[valid, :]

    # Auto-select shrinkage
    if shrink == "auto":
        if n < 10:
            shrink = "none"
            warnings.warn(
                f"Small sample size (n={n}). Using shrink='none' to\
                 avoid over-shrinkage."
            )
        elif n < 30:
            log_s2 = np.log(s2[np.isfinite(s2)])
            var_log_s2 = np.var(log_s2, ddof=1)
            if var_log_s2 < 0.05:
                shrink = 10.0
                warnings.warn(
                    f"Low variance heterogeneity (Var[log(s²)]={var_log_s2:.4f}). "
                    f"Using fixed shrink={shrink} for speed."
                )
            else:
                shrink = "smyth"
        else:
            shrink = "smyth"

    # Apply shrinkage
    if isinstance(shrink, (int, float)) and shrink > 0:
        # Fixed shrinkage
        d0 = float(min(shrink, max_d0))
        if robust:
            s2_for_target = _winsorize_array(s2, winsor_lower, winsor_upper)
        else:
            s2_for_target = s2
        s0sq = float(np.median(s2_for_target))
        s2_post = (df_resid * s2 + d0 * s0sq) / (df_resid + d0)
        df_total = df_resid + d0

    elif shrink == "median":
        # Median-based shrinkage
        if robust:
            s2_for_target = _winsorize_array(s2, winsor_lower, winsor_upper)
            s0sq = float(np.median(s2_for_target))
        else:
            s0sq = float(np.median(s2))
        d0 = float(max(2.0, min(max_d0, n / 2.0)))
        s2_post = (df_resid * s2 + d0 * s0sq) / (df_resid + d0)
        df_total = df_resid + d0

    elif shrink == "smyth":
        # Smyth's empirical Bayes
        d0, s0sq = _estimate_smyth_prior(s2, df_resid, robust=robust, max_d0=max_d0)
        s2_post = (df_resid * s2 + d0 * s0sq) / (df_resid + d0)
        df_total = df_resid + d0

    elif shrink == "none":
        # No shrinkage
        d0 = 0.0
        s2_post = s2.copy()
        df_total = df_resid

    else:
        raise ValueError(
            f"Unsupported shrink option: {shrink}. "
            "Use 'auto', 'smyth', 'median', 'none', or a numeric value."
        )

    XtX = X.T @ X
    XtX_inv = linalg.pinv(XtX)

    # Single contrast test
    if contrast is not None:
        contrast = np.asarray(contrast).reshape(-1)

        logFC = contrast @ beta_hat
        cc = contrast @ XtX_inv @ contrast
        se = np.sqrt(np.maximum(cc * s2, eps))
        se_post = np.sqrt(np.maximum(cc * s2_post, eps))
        t_stat = logFC / se_post

        pvals = 2.0 * stats.t.sf(np.abs(t_stat), df=df_total)
        reject, padj, _, _ = multipletests(pvals, method="fdr_bh")

        res = pd.DataFrame(
            {
                "logFC": logFC,
                "se": se,
                "t": t_stat,
                "pval": pvals,
                "padj": padj,
                "df_resid": df_resid,
                "df_total": df_total,
                "s2": s2,
                "s2_post": s2_post,
                "d0": d0,
                "n_obs": n_obs[valid],
            },
            index=M_valid.index,
        )
        res = res.sort_values("pval")

    # Multi-coefficient F-test
    elif contrast_matrix is not None:
        R = np.asarray(contrast_matrix)
        if R.ndim != 2 or R.shape[1] != p:
            raise ValueError("contrast_matrix must have shape (r, p)")

        r = np.linalg.matrix_rank(R)
        RVR = R @ XtX_inv @ R.T
        RVR_inv = linalg.pinv(RVR)
        CB = R @ beta_hat
        CBt = CB.T
        temp = CBt @ RVR_inv
        numer = np.sum(temp * CBt, axis=1)

        F_stat = (numer / r) / np.maximum(s2_post, eps)
        pvals = stats.f.sf(F_stat, r, df_total)
        reject, padj, _, _ = multipletests(pvals, method="fdr_bh")

        res = pd.DataFrame(
            {
                "F": F_stat,
                "pval": pvals,
                "padj": padj,
                "df1": r,
                "df2": df_total,
                "s2": s2,
                "s2_post": s2_post,
                "d0": d0,
                "n_obs": n_obs[valid],
            },
            index=M_valid.index,
        )
        res = res.sort_values("pval")

    # No contrast (just return variance estimates)
    else:
        res = pd.DataFrame(
            {
                "df_resid": df_resid,
                "s2": s2,
                "s2_post": s2_post,
                "d0": d0,
                "n_obs": n_obs[valid],
            },
            index=M_valid.index,
        )

    # Add group means
    res = _add_group_means(res, M_valid, design, add_beta=True)

    if return_residuals:
        resid_df = pd.DataFrame(residuals_valid, index=M_valid.index, columns=M.columns)
        return res, resid_df

    return res


def fit_differential_chunked(
    M: pd.DataFrame,
    design: pd.DataFrame,
    chunk_size: int = 10000,
    verbose: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """
    Memory-efficient differential analysis for large datasets.

    Processes CpGs in chunks to avoid memory issues with 450K/EPIC arrays.

    Parameters
    ----------
    M : pd.DataFrame
        Large CpG x samples matrix (e.g., 450K = ~450,000 CpGs)
    design : pd.DataFrame
        Design matrix
    chunk_size : int
        Number of CpGs to process per chunk (default: 10000)
    verbose : bool
        Print progress messages
    **kwargs
        Additional arguments passed to fit_differential

    Returns
    -------
    pd.DataFrame
        Combined results with re-adjusted p-values

    Examples
    --------
    >>> # For 450K array with 450,000 CpGs
    >>> results = fit_differential_chunked(
    ...     M_450k,
    ...     design,
    ...     chunk_size=10000,
    ...     contrast=np.array([0, 1]),
    ...     shrink=10.0  # Fixed shrinkage for speed
    ... )
    """
    n_chunks = int(np.ceil(len(M) / chunk_size))
    results = []

    if verbose:
        print(f"Processing {len(M):,} CpGs in {n_chunks} chunks of {chunk_size:,}...")

    start_time = time.time()

    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(M))

        M_chunk = M.iloc[start_idx:end_idx]

        try:
            res_chunk = fit_differential(M_chunk, design, **kwargs)
            results.append(res_chunk)

            if verbose and (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (n_chunks - i - 1) / rate
                print(
                    f"  Chunk {i+1}/{n_chunks} | "
                    f"{elapsed:.1f}s elapsed | ETA: {eta:.1f}s"
                )

        except Exception as e:
            warnings.warn(f"Chunk {i+1} failed: {e}")
            continue

    if len(results) == 0:
        raise ValueError("All chunks failed to process")

    # Combine results
    combined = pd.concat(results, axis=0)

    # CRITICAL: Re-adjust p-values across ALL CpGs
    if "pval" in combined.columns:
        if verbose:
            print(f"  Re-adjusting p-values across {len(combined):,} CpGs...")
        _, padj_new, _, _ = multipletests(combined["pval"], method="fdr_bh")
        combined["padj"] = padj_new
        combined = combined.sort_values("pval")

    # FIXED: Add group means using FULL data for all result CpGs
    combined = _add_group_means(
        combined,
        M.loc[combined.index],  # FIX: Use full M, subset to result CpGs
        design,
        add_beta=True,
    )

    elapsed = time.time() - start_time
    if verbose:
        print(f"✔ Completed in {elapsed:.1f}s ({len(combined)/elapsed:.0f} CpGs/sec)")

    return combined


# ============================================================================
# DATA PREPROCESSING
# ============================================================================

def filter_cpgs_by_missingness(
    M: pd.DataFrame,
    max_missing_rate: float = 0.2,
    min_samples_per_group: Optional[int] = None,
    groups: Optional[pd.Series] = None,
) -> Tuple[pd.DataFrame, int, int]:
    """
    Filter CpGs based on missing data thresholds (optimized).

    Parameters
    ----------
    M : pd.DataFrame
        CpG x samples matrix
    max_missing_rate : float
        Maximum fraction of missing values per CpG (0-1)
    min_samples_per_group : int, optional
        Minimum non-missing samples required per group
    groups : pd.Series, optional
        Sample group labels (required if min_samples_per_group specified)

    Returns
    -------
    Tuple[pd.DataFrame, int, int]
        (filtered_M, n_filtered, n_retained)
    """
    # Convert to numpy for speed
    M_values = M.values
    n_cpgs, n_samples = M_values.shape
    
    # Vectorized: missing rate filter
    n_missing = np.isnan(M_values).sum(axis=1)
    keep = (n_missing / n_samples) <= max_missing_rate

    if min_samples_per_group is not None:
        if groups is None:
            raise ValueError(
                "`groups` must be provided if `min_samples_per_group` is specified."
            )
        
        # Align groups with M columns
        groups_aligned = groups.loc[M.columns].values
        
        # Get unique groups and create masks
        unique_groups = np.unique(groups_aligned)
        
        # Pre-compute non-missing mask (boolean array)
        not_missing = ~np.isnan(M_values)
        
        # Vectorized group filtering
        for group in unique_groups:
            group_indices = np.where(groups_aligned == group)[0]
            # Count non-missing in this group for all CpGs at once
            n_present = not_missing[:, group_indices].sum(axis=1)
            keep &= (n_present >= min_samples_per_group)

    filtered_M = M[keep]
    n_filtered = int((~keep).sum())
    n_retained = int(keep.sum())
    return filtered_M, n_filtered, n_retained
    

def impute_missing_values_fast(
    M: pd.DataFrame,
    method: str = "mean",
    k: int = 5,
    use_sample_knn: bool = False
) -> pd.DataFrame:
    """
    Fast imputation for methylation matrices (CpGs x samples).

    Parameters
    ----------
    M : pd.DataFrame
        CpG x samples matrix with missing values
    method : str
        'mean', 'median', or 'knn'
    k : int
        Number of neighbors for KNN
    use_sample_knn : bool
        If True, performs KNN across samples (fast if samples << CpGs)

    Returns
    -------
    pd.DataFrame
        Fully imputed matrix
    """
    M_copy = M.copy()

    if method in ("mean", "median"):
        # Row-wise fast imputation using numpy
        if method == "mean":
            fill_values = np.nanmean(M_copy.values, axis=1)
        else:
            fill_values = np.nanmedian(M_copy.values, axis=1)
        
        # Broadcast fill_values to match shape and fill NaNs
        mask = np.isnan(M_copy.values)
        M_copy.values[mask] = np.take(fill_values, np.where(mask)[0])
        return M_copy

    elif method == "knn":
        if use_sample_knn:
            # KNN across samples (columns) - fast for hundreds of samples
            M_filled = M_copy.T  # now samples x CpGs
            imputer = KNNImputer(n_neighbors=k, weights="distance")
            M_imputed = imputer.fit_transform(M_filled)
            return pd.DataFrame(M_imputed.T, index=M_copy.index, columns=M_copy.columns)
        else:
            # Naive row-wise KNN (slow for >100k rows)
            imputer = KNNImputer(n_neighbors=k, weights="distance")
            M_imputed = imputer.fit_transform(M_copy)
            return pd.DataFrame(M_imputed, index=M_copy.index, columns=M_copy.columns)
    else:
        raise ValueError(f"Unknown method: {method}")


def filter_min_per_group(
    M: pd.DataFrame,
    groups: pd.Series,
    min_per_group: int = 5,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Remove CpGs that do not have at least min_per_group non-missing
    values in EVERY group.

    Parameters
    ----------
    M : pd.DataFrame
        CpG × samples matrix (may contain NaN)
    groups : pd.Series
        Sample → group label (index must match M.columns)
    min_per_group : int
        Minimum number of observations required in each group
    verbose : bool
        Print summary

    Returns
    -------
    pd.DataFrame
        Subset of M that passes the filter
    """
    groups = groups.loc[M.columns]
    
    # Create boolean mask of non-missing values
    not_missing = ~M.isna()
    
    # Count non-missing values per group for each CpG
    # This is vectorized across all CpGs at once
    counts_per_group = {}
    for group_label in groups.unique():
        group_mask = groups == group_label
        counts_per_group[group_label] = not_missing.loc[:, group_mask].sum(axis=1)
    
    # Convert to DataFrame for easy filtering
    counts_df = pd.DataFrame(counts_per_group)
    
    # Keep CpGs where ALL groups have >= min_per_group observations
    keep_mask = (counts_df >= min_per_group).all(axis=1)
    
    if verbose:
        n_kept = keep_mask.sum()
        n_removed = len(M) - n_kept
        print(
            f"filter_min_per_group: kept {n_kept:,} / {len(M):,} "
            f"CpGs (removed {n_removed:,} with <{min_per_group} obs per group)"
        )
    
    return M.loc[keep_mask]


# ============================================================================
# RESULTS ANALYSIS
# ============================================================================


def summarize_differential_results(
    res: pd.DataFrame, pval_thresh: float = 0.05
) -> Dict:
    """Generate comprehensive summary statistics."""
    sig = res[res["padj"] < pval_thresh]

    summary = {
        "total_tested": len(res),
        "significant": len(sig),
        "pct_significant": len(sig) / len(res) * 100 if len(res) > 0 else 0,
        "hypermethylated": (
            len(sig[sig["logFC"] > 0]) if "logFC" in sig.columns else 0
        ),
        "hypomethylated": (len(sig[sig["logFC"] < 0]) if "logFC" in sig.columns else 0),
        "mean_logFC_sig": (
            sig["logFC"].abs().mean() if len(sig) > 0 and "logFC" in sig.columns else 0
        ),
        "median_logFC_sig": (
            sig["logFC"].abs().median()
            if len(sig) > 0 and "logFC" in sig.columns
            else 0
        ),
        "max_abs_logFC": (res["logFC"].abs().max() if "logFC" in res.columns else 0),
        "min_pval": res["pval"].min() if "pval" in res.columns else 1,
        "shrinkage_factor": (
            res["s2_post"].median() / res["s2"].median() if "s2" in res.columns else 1
        ),
        "d0": res["d0"].iloc[0] if "d0" in res.columns else 0,
    }

    return summary


def get_significant_cpgs(
    res: pd.DataFrame,
    lfc_col: str = "logFC",
    pval_col: str = "padj",
    lfc_thresh: float = 0,
    pval_thresh: float = 0.05,
    delta_beta_thresh: Optional[float] = None,
    direction: Optional[str] = None,
    return_summary: bool = False,
) -> Union[List[str], Dict]:
    """
    Extract significant CpGs with optional delta-beta threshold.

    Parameters
    ----------
    res : pd.DataFrame
        Results from fit_differential
    lfc_col : str
        Column name for log fold change
    pval_col : str
        Column name for adjusted p-value
    lfc_thresh : float
        Minimum absolute log fold change
    pval_thresh : float
        Maximum adjusted p-value
    delta_beta_thresh : float, optional
        Minimum absolute beta-value difference
    direction : str, optional
        'hyper', 'hypo', or None for both
    return_summary : bool
        Return dictionary with summary statistics

    Returns
    -------
    List[str] or Dict
        List of CpG IDs or summary dictionary
    """
    df = res.copy()
    sig = df[df[pval_col] < pval_thresh]

    if direction == "hyper":
        sig = sig[sig[lfc_col] >= lfc_thresh]
    elif direction == "hypo":
        sig = sig[sig[lfc_col] <= -lfc_thresh]
    else:
        sig = sig[abs(sig[lfc_col]) >= lfc_thresh]

    if delta_beta_thresh is not None:
        beta_cols = [c for c in sig.columns if c.startswith("meanB_")]
        if len(beta_cols) >= 2:
            delta_beta = abs(sig[beta_cols[0]] - sig[beta_cols[1]])
            sig = sig[delta_beta >= delta_beta_thresh]

    cpg_list = sig.index.tolist()

    if return_summary:
        summary = {
            "n_significant": len(cpg_list),
            "n_hyper": len(sig[sig[lfc_col] > 0]),
            "n_hypo": len(sig[sig[lfc_col] < 0]),
            "mean_abs_lfc": abs(sig[lfc_col]).mean(),
            "cpgs": cpg_list,
        }
        return summary

    return cpg_list


def export_results(
    res: pd.DataFrame,
    output_path: str,
    format: str = "csv",
    include_all: bool = False,
):
    """Export results to file."""

    if not include_all:
        cols = ["logFC", "t", "pval", "padj", "meanM_Normal", "meanM_Tumor"]
        res_export = res[[c for c in cols if c in res.columns]]
    else:
        res_export = res

    if format == "csv":
        res_export.to_csv(output_path)
    elif format == "excel":
        res_export.to_excel(output_path, engine="openpyxl")
    elif format == "tsv":
        res_export.to_csv(output_path, sep="\t")
    else:
        raise ValueError(f"Unsupported format: {format}")

    print(f"✔ Results exported to {output_path}")


# ============================================================================
# VISUALIZATION
# ============================================================================


def plot_mean_variance(
    res: pd.DataFrame, save_path: Optional[str] = None, dpi: int = 300
):
    """Plot mean-variance relationship (SA plot)."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    log_s2 = np.log2(res["s2"])
    log_s2_post = np.log2(res["s2_post"])
    mean_expr = res[[c for c in res.columns if c.startswith("meanM_")]].mean(axis=1)

    ax1.scatter(mean_expr, log_s2, alpha=0.3, s=10, label="Raw variance")
    ax1.scatter(mean_expr, log_s2_post, alpha=0.3, s=10, label="Moderated variance")
    ax1.set_xlabel("Mean M-value")
    ax1.set_ylabel("log2(variance)")
    ax1.set_title("Mean-Variance Relationship")
    ax1.legend()
    ax1.grid(alpha=0.3)

    rank = np.arange(len(res))
    ax2.scatter(rank, np.sqrt(res["s2"]), alpha=0.3, s=10, label="Raw SD")
    ax2.scatter(rank, np.sqrt(res["s2_post"]), alpha=0.3, s=10, label="Moderated SD")
    ax2.set_xlabel("Rank")
    ax2.set_ylabel("Standard Deviation")
    ax2.set_title("Variance Shrinkage")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()

    return fig


def plot_residual_diagnostics(
    residuals: pd.DataFrame,
    M: pd.DataFrame,
    design: pd.DataFrame,
    top_n: int = 9,
    save_path: Optional[str] = None,
    dpi: int = 300,
):
    """Plot residual diagnostics for top variable CpGs."""
    import matplotlib.pyplot as plt

    var_cpgs = residuals.var(axis=1).nlargest(top_n).index

    n_rows = int(np.ceil(np.sqrt(top_n)))
    n_cols = int(np.ceil(top_n / n_rows))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = axes.flatten() if top_n > 1 else [axes]

    for idx, cpg in enumerate(var_cpgs):
        ax = axes[idx]
        resid = residuals.loc[cpg].dropna()
        stats.probplot(resid, dist="norm", plot=ax)
        ax.set_title(f"{cpg}", fontsize=10)
        ax.grid(alpha=0.3)

    for idx in range(len(var_cpgs), len(axes)):
        axes[idx].axis("off")

    plt.suptitle("Residual Q-Q Plots (Top Variable CpGs)", fontsize=14, y=1.00)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()

    return fig


def plot_sample_qc(
    M: pd.DataFrame,
    metadata: pd.DataFrame,
    group_col: str = "Type",
    save_path: Optional[str] = None,
    dpi: int = 300,
):
    """Sample-level QC metrics visualization."""
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Missing data
    missing_per_sample = M.isna().sum(axis=0) / len(M) * 100
    ax = axes[0, 0]
    colors = ["skyblue" if g == "Normal" else "salmon" for g in metadata[group_col]]
    ax.bar(
        range(len(missing_per_sample)),
        missing_per_sample.values,
        color=colors,
    )
    ax.set_xlabel("Sample")
    ax.set_ylabel("% Missing")
    ax.set_title("Missing Data per Sample")
    ax.axhline(10, color="red", linestyle="--", alpha=0.5, label="10% threshold")
    ax.legend()

    # Mean M-value
    ax = axes[0, 1]
    mean_M = M.mean(axis=0)
    for group in metadata[group_col].unique():
        mask = metadata[group_col] == group
        ax.hist(mean_M[mask], alpha=0.6, bins=20, label=group)
    ax.set_xlabel("Mean M-value")
    ax.set_ylabel("Frequency")
    ax.set_title("Mean M-value Distribution by Group")
    ax.legend()

    # Variance
    ax = axes[1, 0]
    var_M = M.var(axis=0)
    colors = ["skyblue" if g == "Normal" else "salmon" for g in metadata[group_col]]
    ax.bar(range(len(var_M)), var_M.values, color=colors)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Variance")
    ax.set_title("Within-Sample Variance")

    # PCA (handle missing data)
    ax = axes[1, 1]
    M_for_pca = M.dropna()
    if len(M_for_pca) < 100:
        # Too few complete CpGs, impute
        from sklearn.impute import SimpleImputer

        imputer = SimpleImputer(strategy="mean")
        M_imputed = imputer.fit_transform(M.T)
        pca = PCA(n_components=2)
        coords = pca.fit_transform(M_imputed)
    else:
        pca = PCA(n_components=2)
        coords = pca.fit_transform(M_for_pca.T)

    for group in metadata[group_col].unique():
        mask = metadata[group_col] == group
        color = "skyblue" if group == "Normal" else "salmon"
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            label=group,
            alpha=0.7,
            s=80,
            color=color,
            edgecolor="k",
        )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("PCA of Samples")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()

    return fig


def plot_pvalue_qq(res: pd.DataFrame, save_path: Optional[str] = None, dpi: int = 300):
    """Q-Q plot of p-values to check for inflation."""
    import matplotlib.pyplot as plt

    observed = -np.log10(np.sort(res["pval"].values))
    expected = -np.log10(np.linspace(1 / len(res), 1, len(res)))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(expected, observed, alpha=0.5, s=10)
    ax.plot([0, max(expected)], [0, max(expected)], "r--", lw=2, label="Expected")

    ax.set_xlabel("Expected -log10(p)", fontsize=12)
    ax.set_ylabel("Observed -log10(p)", fontsize=12)
    ax.set_title("P-value Q-Q Plot", fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()

    return fig


def plot_volcano_enhanced(
    res: pd.DataFrame,
    lfc_col: str = "logFC",
    pval_col: str = "pval",
    alpha: float = 0.7,
    lfc_thresh: float = 1,
    pval_thresh: float = 0.05,
    top_n: int = 10,
    save_path: Optional[str] = None,
    dpi: int = 300,
):
    """Enhanced volcano plot with top hit labels."""
    import matplotlib.pyplot as plt

    res = res.copy()
    res["neg_log10_p"] = -np.log10(res[pval_col].replace(0, np.nextafter(0, 1)))

    conditions = [
        (res[pval_col] < pval_thresh) & (res[lfc_col] >= lfc_thresh),
        (res[pval_col] < pval_thresh) & (res[lfc_col] <= -lfc_thresh),
    ]
    choices = ["Hypermethylated", "Hypomethylated"]
    res["Group"] = np.select(conditions, choices, default="Not significant")

    color_map = {
        "Hypermethylated": "red",
        "Hypomethylated": "blue",
        "Not significant": "grey",
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    for group, color in color_map.items():
        subset = res[res["Group"] == group]
        ax.scatter(
            subset[lfc_col],
            subset["neg_log10_p"],
            c=color,
            alpha=alpha,
            edgecolor="k",
            linewidth=0.3,
            s=30,
            label=group,
        )

    ax.axvline(-lfc_thresh, color="black", linestyle="--", lw=1, alpha=0.5)
    ax.axvline(lfc_thresh, color="black", linestyle="--", lw=1, alpha=0.5)
    ax.axhline(-np.log10(pval_thresh), color="black", linestyle="--", lw=1, alpha=0.5)

    # Label top hits
    if top_n > 0:
        sig_hits = res[res["Group"] != "Not significant"].nsmallest(top_n, pval_col)
        for idx, row in sig_hits.iterrows():
            ax.annotate(
                idx,
                xy=(row[lfc_col], row["neg_log10_p"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=7,
                alpha=0.7,
            )

    ax.set_xlabel("Log2 Fold Change", fontsize=12)
    ax.set_ylabel("-log10(p-value)", fontsize=12)
    ax.set_title("Volcano Plot: Differential Methylation", fontsize=14)
    ax.legend(loc="upper center")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()

    return fig


# ============================================================================
# PDF REPORTING
# ============================================================================


class PDFLogger:
    """PDF logger for Markdown-like reports."""

    def __init__(self, path: Optional[str] = "log.pdf", echo: bool = True):
        self.path = path
        self.echo = echo
        self.doc = SimpleDocTemplate(path, pagesize=A4)
        self.styles = getSampleStyleSheet()

        self.styles.add(
            ParagraphStyle(
                "H1",
                parent=self.styles["Normal"],
                fontName="Helvetica-Bold",
                fontSize=16,
                leading=18,
                spaceBefore=25,
                spaceAfter=6,
            )
        )
        self.styles.add(
            ParagraphStyle(
                "H2",
                parent=self.styles["Normal"],
                fontName="Helvetica-Bold",
                fontSize=14,
                leading=16,
                spaceBefore=20,
                spaceAfter=6,
            )
        )
        self.styles.add(
            ParagraphStyle(
                "H3",
                parent=self.styles["Normal"],
                fontName="Helvetica-Bold",
                fontSize=12,
                leading=14,
                spaceBefore=6,
                spaceAfter=6,
            )
        )
        self.styles.add(
            ParagraphStyle(
                "CodeBlock",
                parent=self.styles["Normal"],
                fontName="Courier",
                fontSize=9,
            )
        )

        self.story: list[Any] = []
        self.current_list = None

    def _format_md(self, text: str) -> str:
        """Basic Markdown formatting: bold, italics, inline code"""
        text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
        text = re.sub(r'\*([^*]+)\*', r'<i>\1</i>', text)
        text = re.sub(r"`(.*?)`", r"<font name='Courier'>\1</font>", text)
        return text

    def _echo(self, text: str):
        if self.echo:
            print(text)

    def _flush_list(self):
        """Close any currently open bullet list."""
        self.current_list = None

    def log_text(self, text: str):
        text = text.strip()
        if not text:
            return
        self._echo(text)

        if text.startswith("### "):
            self._flush_list()
            self.story.append(Paragraph(text[4:], self.styles["H3"]))
        elif text.startswith("## "):
            self._flush_list()
            self.story.append(Paragraph(text[3:], self.styles["H2"]))
        elif text.startswith("# "):
            self._flush_list()
            self.story.append(Paragraph(text[2:], self.styles["H1"]))
        elif text.startswith("- "):
            item = ListItem(
                Paragraph(
                    self._format_md(text[2:].strip()).rstrip("\n"),
                    self.styles["Normal"],
                )
            )
            if self.current_list is None:
                self.current_list = ListFlowable(
                    [item],
                    bulletType="bullet",
                    leftIndent=18,
                    bulletFontSize=10,
                    start=None,
                    spaceBefore=0,
                    spaceAfter=12,
                )
                self.story.append(self.current_list)
            else:
                self.current_list._flowables.append(item)
        else:
            self._flush_list()
            html = self._format_md(text)
            self.story.append(Paragraph(html, self.styles["Normal"]))

        if self.current_list is None:
            self.story.append(Spacer(1, 0 * inch))

    def log_code(self, code: str):
        """Log preformatted code block"""
        code = code.rstrip()
        self._echo(code)
        self.story.append(Preformatted(code, self.styles["CodeBlock"]))
        self.story.append(Spacer(1, 0.18 * inch))

    def log_dataframe(
        self, df: pd.DataFrame, title: Optional[str] = None, max_rows: int = 10
    ):
        """Log a pandas DataFrame as a formatted table"""
        self._flush_list()

        if title:
            self.story.append(Paragraph(f"<b>{title}</b>", self.styles["Normal"]))
            self.story.append(Spacer(1, 0.1 * inch))

        if len(df) > max_rows:
            df_display = pd.concat([df.head(max_rows // 2), df.tail(max_rows // 2)])
            table_text = df_display.to_string()
            table_text += f"\n... ({len(df) - max_rows} more rows)"
        else:
            table_text = df.to_string()

        self._echo(table_text)
        self.story.append(Preformatted(table_text, self.styles["CodeBlock"]))
        self.story.append(Spacer(1, 0.15 * inch))

    def log_image(
        self,
        path: str,
        title: Optional[str] = None,
        caption: Optional[str] = None,
        width: float = 5 * inch,
    ):
        """Add an image to the PDF with auto-scaling"""
        self._echo(f"[Image: {path}] {caption or ''}")
        if not os.path.exists(path):
            self.log_text(f"[Missing image: {path}]")
            return

        img = ImageReader(path)
        iw, ih = img.getSize()
        aspect = ih / float(iw)
        height = width * aspect

        max_height = 9 * inch
        if height > max_height:
            height = max_height
            width = height / aspect

        max_width = A4[0] - 2 * inch
        if width > max_width:
            width = max_width
            height = width * aspect

        self.story.append(Image(path, width=width, height=height))
        if caption or title:
            caption_text = f"<b>{title}</b>: {caption}" if title else caption
            self.story.append(Paragraph(caption_text, self.styles["Normal"]))
        self.story.append(Spacer(1, 0.2 * inch))

    def save(self):
        """Build the PDF"""
        try:
            self.doc.build(self.story)
            self._echo(f"✔ PDF saved to {self.path}")
        except Exception as e:
            self._echo(f"! PDF build failed: {e}")
