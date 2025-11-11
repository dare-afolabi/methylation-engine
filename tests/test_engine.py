#!/usr/bin/env python
# coding: utf-8

"""
Comprehensive test suite for differential methylation analysis engine
Target: 100% code coverage

Run with:
    pytest tests/test_engine.py -v --cov=core.engine --cov-report=html
"""

# import os
# import tempfile
# import warnings
from io import StringIO
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from core.engine import (
    PDFLogger,
    _add_group_means,
    _estimate_smyth_prior,
    _winsorize_array,
    export_results,
    filter_cpgs_by_missingness,
    filter_min_per_group,
    fit_differential,
    fit_differential_chunked,
    get_significant_cpgs,
    impute_missing_values_fast,
    plot_mean_variance,
    plot_pvalue_qq,
    plot_residual_diagnostics,
    plot_sample_qc,
    plot_volcano_enhanced,
    summarize_differential_results,
    validate_contrast,
    validate_design,
)

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def simple_data():
    """Create simple test dataset."""
    np.random.seed(1500)
    n_cpg, n_samples = 100, 10

    M = pd.DataFrame(
        np.random.randn(n_cpg, n_samples),
        index=[f"cg{i:08d}" for i in range(n_cpg)],
        columns=[f"S{i}" for i in range(n_samples)],
    )

    metadata = pd.DataFrame({"Type": ["Normal"] * 5 + ["Tumor"] * 5}, index=M.columns)

    design = pd.DataFrame({"Intercept": 1, "Group": [0] * 5 + [1] * 5}, index=M.columns)

    return M, metadata, design


@pytest.fixture
def data_with_missing():
    """Create dataset with missing values."""
    np.random.seed(1500)
    M = pd.DataFrame(
        np.random.randn(50, 8),
        columns=[f"S{i}" for i in range(8)],
        index=[f"cg{i:08d}" for i in range(50)],
    )

    # Add missing values
    M.iloc[0:5, 0:2] = np.nan
    M.iloc[10:15, 3:5] = np.nan

    metadata = pd.DataFrame({"Type": ["Normal"] * 4 + ["Tumor"] * 4}, index=M.columns)

    return M, metadata


@pytest.fixture
def large_data():
    """Create large dataset for chunking tests."""
    np.random.seed(1500)
    n_cpg, n_samples = 5000, 12

    M = pd.DataFrame(
        np.random.randn(n_cpg, n_samples),
        index=[f"cg{i:08d}" for i in range(n_cpg)],
        columns=[f"S{i}" for i in range(n_samples)],
    )

    # Add differential methylation
    M.iloc[0:50, 6:] += 2.0

    design = pd.DataFrame({"Intercept": 1, "Group": [0] * 6 + [1] * 6}, index=M.columns)

    return M, design


# ============================================================================
# VALIDATION TESTS
# ============================================================================


class TestValidation:
    """Test input validation functions."""

    def test_validate_design_correct(self, simple_data):
        """Valid design should pass."""
        M, _, design = simple_data
        validate_design(design, M)  # Should not raise

    def test_validate_design_not_dataframe(self, simple_data):
        """Non-DataFrame design should raise TypeError."""
        M, _, _ = simple_data
        with pytest.raises(TypeError, match="design must be a pandas DataFrame"):
            validate_design([[1, 0]] * 10, M)

    def test_validate_design_wrong_length(self):
        """Mismatched design length should raise ValueError."""
        M = pd.DataFrame(np.random.randn(10, 5), columns=[f"S{i}" for i in range(5)])
        design = pd.DataFrame({"Group": [0, 1, 0]})

        with pytest.raises(ValueError, match="design rows"):
            validate_design(design, M)

    def test_validate_design_mismatched_index(self, simple_data):
        """Mismatched indices should raise ValueError."""
        M, _, design = simple_data
        design.index = [f"X{i}" for i in range(len(design))]

        with pytest.raises(ValueError, match="design index must exactly match"):
            validate_design(design, M)

    def test_validate_design_missing_values(self, simple_data):
        """Design with NaN should raise ValueError."""
        M, _, design = simple_data
        design.iloc[0, 0] = np.nan

        with pytest.raises(ValueError, match="design contains missing values"):
            validate_design(design, M)

    def test_validate_design_non_numeric(self, simple_data):
        """Non-numeric design should raise ValueError."""
        M, _, design = simple_data
        design["String"] = ["A"] * len(design)

        with pytest.raises(ValueError, match="non-numeric columns"):
            validate_design(design, M)

    def test_validate_design_too_many_covariates(self):
        """Too many covariates should raise ValueError."""
        M = pd.DataFrame(np.random.randn(10, 5), columns=[f"S{i}" for i in range(5)])
        design = pd.DataFrame(
            {
                f"Cov{i}": np.random.randn(5)
                for i in range(6)  # 6 covariates for 5 samples
            },
            index=M.columns,
        )

        with pytest.raises(ValueError, match="Too many covariates"):
            validate_design(design, M)

    def test_validate_contrast_correct(self, simple_data):
        """Valid contrast should pass."""
        _, _, design = simple_data
        contrast = np.array([0, 1])
        validate_contrast(contrast, design)  # Should not raise

    def test_validate_contrast_wrong_length(self, simple_data):
        """Wrong length contrast should raise ValueError."""
        _, _, design = simple_data
        contrast = np.array([0, 1, 0])  # Wrong length

        with pytest.raises(ValueError, match="Contrast length"):
            validate_contrast(contrast, design)

    def test_validate_contrast_non_finite(self, simple_data):
        """Non-finite contrast should raise ValueError."""
        _, _, design = simple_data
        contrast = np.array([0, np.nan])

        with pytest.raises(ValueError, match="non-finite values"):
            validate_contrast(contrast, design)


# ============================================================================
# CORE FUNCTION TESTS
# ============================================================================


class TestCoreFunctions:
    """Test core statistical functions."""

    def test_winsorize_array(self):
        """Test winsorization."""
        x = np.array([1, 2, 3, 4, 5, 100, 200])
        result = _winsorize_array(x, lower_q=0.1, upper_q=0.9)

        # Extreme values should be clipped
        assert result.max() < 200
        assert result.min() > 1
        assert len(result) == len(x)

    def test_winsorize_array_with_nan(self):
        """Test winsorization handles NaN."""
        x = np.array([1, 2, np.nan, 4, 5, 100])
        result = _winsorize_array(x)

        # Should not crash, NaN handled by nanquantile
        assert not np.isnan(result[~np.isnan(x)]).any()

    def test_estimate_smyth_prior_normal(self):
        """Test Smyth prior estimation with normal heterogeneity."""
        s2 = np.random.gamma(2, 2, 1000)  # Varied variances
        df_resid = 10

        d0, s0sq = _estimate_smyth_prior(s2, df_resid, robust=True)

        assert d0 > 0
        assert d0 <= 50  # max_d0 default
        assert s0sq > 0

    def test_estimate_smyth_prior_low_heterogeneity(self):
        """Test Smyth prior with low variance heterogeneity."""
        s2 = np.ones(100) * 2.0  # Very similar variances
        df_resid = 10

        with pytest.warns(UserWarning, match="Variance heterogeneity is low"):
            d0, s0sq = _estimate_smyth_prior(s2, df_resid)

        assert d0 == 10.0  # Should use conservative value
        assert s0sq > 0

    def test_estimate_smyth_prior_no_finite_variance(self):
        """Test error when no finite variances."""
        s2 = np.array([np.nan, np.inf, -np.inf])

        with pytest.raises(ValueError, match="No finite variances"):
            _estimate_smyth_prior(s2, df_resid=10)

    def test_add_group_means_missing_cpgs(self, simple_data):
        """Test error when M_df missing CpGs."""
        M, _, design = simple_data

        res = pd.DataFrame(
            {"logFC": [1, 2, 3]}, index=["cg_fake1", "cg_fake2", "cg_fake3"]
        )

        with pytest.raises(ValueError, match="M_df missing"):
            _add_group_means(res, M, design)

    def test_add_group_means_no_suitable_column(self, simple_data):
        """Test error when no suitable grouping column."""
        M, _, _ = simple_data
        design = pd.DataFrame({"AllSame": [1] * len(M.columns)}, index=M.columns)

        res = pd.DataFrame({"logFC": [1]}, index=M.index[:1])

        with pytest.raises(ValueError, match="No suitable grouping column"):
            _add_group_means(res, M, design)


# ============================================================================
# DIFFERENTIAL ANALYSIS TESTS
# ============================================================================


class TestDifferentialAnalysis:
    """Test fit_differential function."""

    def test_fit_differential_basic(self, simple_data):
        """Test basic two-group comparison."""
        M, _, design = simple_data
        contrast = np.array([0, 1])

        results = fit_differential(M, design, contrast=contrast, shrink="none")

        # Check structure
        assert "logFC" in results.columns
        assert "pval" in results.columns
        assert "padj" in results.columns
        assert "t" in results.columns
        assert "se" in results.columns
        assert len(results) == len(M)

        # Check value ranges
        assert results["pval"].min() >= 0
        assert results["pval"].max() <= 1
        assert results["padj"].min() >= 0
        assert results["padj"].max() <= 1

    def test_fit_differential_with_shrinkage_fixed(self, simple_data):
        """Test with fixed shrinkage."""
        M, _, design = simple_data

        results = fit_differential(
            M, design, contrast=np.array([0, 1]), shrink=10.0, robust=True
        )

        assert results["d0"].iloc[0] == 10.0
        assert "s2_post" in results.columns
        assert not np.allclose(results["s2"], results["s2_post"])

    def test_fit_differential_with_shrinkage_median(self, simple_data):
        """Test with median shrinkage."""
        M, _, design = simple_data

        results = fit_differential(
            M, design, contrast=np.array([0, 1]), shrink="median"
        )

        assert "d0" in results.columns
        assert results["d0"].iloc[0] > 0

    def test_fit_differential_with_shrinkage_smyth(self, simple_data):
        """Test with Smyth shrinkage."""
        M, _, design = simple_data

        results = fit_differential(M, design, contrast=np.array([0, 1]), shrink="smyth")

        assert "d0" in results.columns
        assert results["d0"].iloc[0] > 0

    def test_fit_differential_auto_shrinkage_small_n(self):
        """Test auto shrinkage with small sample size."""
        M = pd.DataFrame(np.random.randn(50, 6), columns=[f"S{i}" for i in range(6)])
        design = pd.DataFrame(
            {"Intercept": 1, "Group": [0] * 3 + [1] * 3}, index=M.columns
        )

        with pytest.warns(UserWarning, match="Small sample size"):
            results = fit_differential(
                M, design, contrast=np.array([0, 1]), shrink="auto"
            )

        assert "d0" in results.columns

    def test_fit_differential_invalid_shrinkage(self, simple_data):
        """Test error with invalid shrinkage option."""
        M, _, design = simple_data

        with pytest.raises(ValueError, match="Unsupported shrink option"):
            fit_differential(M, design, contrast=np.array([0, 1]), shrink="invalid")

    def test_fit_differential_with_missing(self, data_with_missing):
        """Test handling of missing data."""
        M, metadata = data_with_missing
        design = pd.DataFrame(
            {"Intercept": 1, "Group": [0] * 4 + [1] * 4}, index=M.columns
        )

        with pytest.warns(UserWarning, match="M contains missing values"):
            results = fit_differential(
                M, design, contrast=np.array([0, 1]), min_count=3
            )

        assert len(results) > 0
        assert results["n_obs"].min() >= 3

    def test_fit_differential_returns_residuals(self, simple_data):
        """Test returning residuals."""
        M, _, design = simple_data

        results, residuals = fit_differential(
            M, design, contrast=np.array([0, 1]), shrink="none", return_residuals=True
        )

        assert isinstance(residuals, pd.DataFrame)
        assert residuals.shape[0] == len(results)
        assert residuals.columns.equals(M.columns)

    def test_fit_differential_f_test(self, simple_data):
        """Test F-test for multiple coefficients."""
        M, _, design = simple_data
        design["Batch"] = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

        R = np.array([[0, 1, 0], [0, 0, 1]])

        results = fit_differential(M, design, contrast_matrix=R, shrink=10.0)

        assert "F" in results.columns
        assert "df1" in results.columns
        assert "df2" in results.columns
        assert results["df1"].iloc[0] == 2

    def test_fit_differential_f_test_invalid_matrix(self, simple_data):
        """Test error with invalid contrast matrix."""
        M, _, design = simple_data

        R = np.array([[0, 1, 0, 1]])  # Wrong shape

        with pytest.raises(ValueError, match="contrast_matrix must have shape"):
            fit_differential(M, design, contrast_matrix=R)

    def test_fit_differential_both_contrasts_error(self, simple_data):
        """Test error when both contrast and contrast_matrix provided."""
        M, _, design = simple_data

        with pytest.raises(
            ValueError, match="Provide either contrast or contrast_matrix"
        ):
            fit_differential(
                M, design, contrast=np.array([0, 1]), contrast_matrix=np.array([[0, 1]])
            )

    def test_fit_differential_no_contrast(self, simple_data):
        """Test without any contrast (just variance estimates)."""
        M, _, design = simple_data

        results = fit_differential(M, design, shrink="none")

        assert "s2" in results.columns
        assert "logFC" not in results.columns
        assert "pval" not in results.columns

    def test_fit_differential_no_valid_cpgs(self):
        """Test error when no CpGs can be fit."""
        M = pd.DataFrame(
            np.nan,
            index=[f"cg{i}" for i in range(10)],
            columns=[f"S{i}" for i in range(6)],
        )
        design = pd.DataFrame(
            {"Intercept": 1, "Group": [0] * 3 + [1] * 3}, index=M.columns
        )

        with pytest.raises(ValueError, match="No CpGs could be fit successfully"):
            fit_differential(M, design, contrast=np.array([0, 1]), min_count=3)


# ============================================================================
# CHUNKED PROCESSING TESTS
# ============================================================================


class TestChunkedProcessing:
    """Test chunked differential analysis."""

    def test_fit_differential_chunked_basic(self, large_data):
        """Test basic chunked processing."""
        M, design = large_data

        results = fit_differential_chunked(
            M,
            design,
            chunk_size=1000,
            contrast=np.array([0, 1]),
            shrink=10.0,
            verbose=False,
        )

        assert len(results) > 0
        assert "logFC" in results.columns
        assert "padj" in results.columns

    def test_fit_differential_chunked_verbose(self, large_data):
        """Test verbose output."""
        M, design = large_data
        M_subset = M.iloc[:500]

        # Capture output
        import sys
        from io import StringIO

        old_stdout = sys.stdout
        sys.stdout = StringIO()

        fit_differential_chunked(
            M_subset,
            design,
            chunk_size=100,
            contrast=np.array([0, 1]),
            shrink=10.0,
            verbose=True,
        )

        output = sys.stdout.getvalue()
        sys.stdout = old_stdout

        assert "Processing" in output
        assert "Completed" in output

    def test_fit_differential_chunked_chunk_failure(self, large_data):
        """Test handling of chunk failure."""
        M, design = large_data

        with patch("core.engine.fit_differential") as mock_fit:
            # First chunk succeeds, second fails, third succeeds
            mock_fit.side_effect = [
                pd.DataFrame({"logFC": [1], "pval": [0.01]}, index=["cg00000000"]),
                Exception("Chunk failed"),
                pd.DataFrame({"logFC": [2], "pval": [0.02]}, index=["cg00000001"]),
            ]

            with pytest.warns(UserWarning, match="Chunk .* failed"):
                results = fit_differential_chunked(
                    M.iloc[:300],
                    design,
                    chunk_size=100,
                    contrast=np.array([0, 1]),
                    verbose=False,
                )

            # Should still return results from successful chunks
            assert len(results) > 0

    def test_fit_differential_chunked_all_fail(self, simple_data):
        """Test error when all chunks fail."""
        M, _, design = simple_data

        with patch("core.engine.fit_differential", side_effect=Exception("All fail")):
            with pytest.raises(ValueError, match="All chunks failed to process"):
                fit_differential_chunked(M, design, chunk_size=50, verbose=False)


# ============================================================================
# PREPROCESSING TESTS
# ============================================================================


class TestPreprocessing:
    """Test data preprocessing functions."""

    def test_filter_cpgs_by_missingness_basic(self, data_with_missing):
        """Test basic CpG filtering."""
        M, metadata = data_with_missing

        M_filtered, n_filtered, n_retained = filter_cpgs_by_missingness(
            M, max_missing_rate=0.2
        )

        assert n_filtered + n_retained == len(M)
        assert len(M_filtered) == n_retained
        assert n_filtered > 0

    def test_filter_cpgs_by_missingness_with_groups(self, data_with_missing):
        """Test filtering with per-group requirements."""
        M, metadata = data_with_missing

        M_filtered, _, _ = filter_cpgs_by_missingness(
            M, max_missing_rate=0.5, min_samples_per_group=2, groups=metadata["Type"]
        )

        # Check that remaining CpGs have at least 2 samples per group
        for cpg in M_filtered.index:
            for group in metadata["Type"].unique():
                group_cols = metadata[metadata["Type"] == group].index
                n_present = (~M_filtered.loc[cpg, group_cols].isna()).sum()
                assert n_present >= 2

    def test_filter_cpgs_by_missingness_no_groups_error(self, data_with_missing):
        """Test error when groups not provided but required."""
        M, _ = data_with_missing

        with pytest.raises(ValueError, match="`groups` must be provided"):
            filter_cpgs_by_missingness(M, min_samples_per_group=2)

    def test_impute_missing_values_knn(self, data_with_missing):
        """Test KNN imputation."""
        M, _ = data_with_missing

        n_missing_before = M.isna().sum().sum()
        M_imputed = impute_missing_values_fast(M, method="knn", k=3)
        n_missing_after = M_imputed.isna().sum().sum()

        assert n_missing_after < n_missing_before

    def test_impute_missing_values_invalid_method(self, data_with_missing):
        """Test error with invalid method."""
        M, _ = data_with_missing

        with pytest.raises(ValueError, match="Unknown imputation method"):
            impute_missing_values_fast(M, method="invalid")

    def test_filter_min_per_group(self, data_with_missing):
        """Test filtering by minimum per group."""
        M, metadata = data_with_missing

        M_filtered = filter_min_per_group(
            M, groups=metadata["Type"], min_per_group=3, verbose=False
        )

        # Check all remaining CpGs have >= 3 obs per group
        for cpg in M_filtered.index:
            for group in metadata["Type"].unique():
                group_cols = metadata[metadata["Type"] == group].index
                n_present = (~M_filtered.loc[cpg, group_cols].isna()).sum()
                assert n_present >= 3

    def test_filter_min_per_group_verbose(self, data_with_missing):
        """Test verbose output."""
        M, metadata = data_with_missing

        import sys

        old_stdout = sys.stdout
        sys.stdout = StringIO()

        filter_min_per_group(M, groups=metadata["Type"], min_per_group=2, verbose=True)

        output = sys.stdout.getvalue()
        sys.stdout = old_stdout

        assert "filter_min_per_group:" in output


# ============================================================================
# RESULTS ANALYSIS TESTS
# ============================================================================


class TestResultsAnalysis:
    """Test results analysis functions."""

    def test_summarize_differential_results(self, large_data):
        """Test results summary."""
        M, design = large_data

        results = fit_differential(M, design, contrast=np.array([0, 1]), shrink=10.0)
        summary = summarize_differential_results(results, pval_thresh=0.05)

        # Check all keys present
        required_keys = [
            "total_tested",
            "significant",
            "pct_significant",
            "hypermethylated",
            "hypomethylated",
            "mean_logFC_sig",
            "median_logFC_sig",
            "max_abs_logFC",
            "min_pval",
            "shrinkage_factor",
            "d0",
        ]
        for key in required_keys:
            assert key in summary

        # Check value validity
        assert summary["total_tested"] == len(results)
        assert summary["significant"] <= summary["total_tested"]
        assert (
            summary["hypermethylated"] + summary["hypomethylated"]
            <= summary["significant"]
        )

    def test_summarize_differential_results_no_significant(self, simple_data):
        """Test summary with no significant results."""
        M, _, design = simple_data

        results = fit_differential(M, design, contrast=np.array([0, 1]), shrink="none")
        summary = summarize_differential_results(
            results, pval_thresh=0.0001
        )  # Very stringent

        assert summary["significant"] >= 0
        assert summary["pct_significant"] >= 0

    def test_get_significant_cpgs_all(self, large_data):
        """Test extracting all significant CpGs."""
        M, design = large_data

        results = fit_differential(M, design, contrast=np.array([0, 1]), shrink=10.0)
        sig_cpgs = get_significant_cpgs(
            results, lfc_thresh=1.0, pval_thresh=0.05, direction=None
        )

        assert isinstance(sig_cpgs, list)
        assert all(cpg in results.index for cpg in sig_cpgs)

        # Verify they meet criteria
        for cpg in sig_cpgs:
            assert results.loc[cpg, "padj"] < 0.05
            assert abs(results.loc[cpg, "logFC"]) >= 1.0

    def test_get_significant_cpgs_hyper(self, large_data):
        """Test extracting hypermethylated CpGs."""
        M, design = large_data

        results = fit_differential(M, design, contrast=np.array([0, 1]), shrink=10.0)
        hyper_cpgs = get_significant_cpgs(
            results, lfc_thresh=1.0, pval_thresh=0.05, direction="hyper"
        )

        for cpg in hyper_cpgs:
            assert results.loc[cpg, "logFC"] >= 1.0
            assert results.loc[cpg, "padj"] < 0.05

    def test_get_significant_cpgs_hypo(self, large_data):
        """Test extracting hypomethylated CpGs."""
        M, design = large_data

        results = fit_differential(M, design, contrast=np.array([0, 1]), shrink=10.0)
        hypo_cpgs = get_significant_cpgs(
            results, lfc_thresh=1.0, pval_thresh=0.05, direction="hypo"
        )

        for cpg in hypo_cpgs:
            assert results.loc[cpg, "logFC"] <= -1.0
            assert results.loc[cpg, "padj"] < 0.05

    def test_get_significant_cpgs_with_delta_beta(self, large_data):
        """Test filtering by delta-beta threshold."""
        M, design = large_data

        results = fit_differential(M, design, contrast=np.array([0, 1]), shrink=10.0)

        sig_cpgs = get_significant_cpgs(
            results, lfc_thresh=0.5, pval_thresh=0.05, delta_beta_thresh=0.1
        )

        # Should have fewer CpGs due to delta-beta filter
        sig_cpgs_no_delta = get_significant_cpgs(
            results, lfc_thresh=0.5, pval_thresh=0.05
        )

        assert len(sig_cpgs) <= len(sig_cpgs_no_delta)

    def test_get_significant_cpgs_with_summary(self, large_data):
        """Test returning summary."""
        M, design = large_data

        results = fit_differential(M, design, contrast=np.array([0, 1]), shrink=10.0)
        summary = get_significant_cpgs(
            results, lfc_thresh=1.0, pval_thresh=0.05, return_summary=True
        )

        assert isinstance(summary, dict)
        assert "n_significant" in summary
        assert "n_hyper" in summary
        assert "n_hypo" in summary
        assert "cpgs" in summary
        assert summary["n_significant"] == len(summary["cpgs"])

    def test_export_results_csv(self, large_data, tmp_path):
        """Test exporting to CSV."""
        M, design = large_data
        results = fit_differential(M, design, contrast=np.array([0, 1]), shrink=10.0)

        output_file = tmp_path / "results.csv"
        export_results(results, str(output_file), format="csv", include_all=False)

        assert output_file.exists()

        # Read back and verify
        df = pd.read_csv(output_file, index_col=0)
        assert len(df) == len(results)

    def test_export_results_tsv(self, large_data, tmp_path):
        """Test exporting to TSV."""
        M, design = large_data
        results = fit_differential(M, design, contrast=np.array([0, 1]), shrink=10.0)

        output_file = tmp_path / "results.tsv"
        export_results(results, str(output_file), format="tsv", include_all=True)

        assert output_file.exists()

    def test_export_results_invalid_format(self, large_data, tmp_path):
        """Test error with invalid format."""
        M, design = large_data
        results = fit_differential(M, design, contrast=np.array([0, 1]), shrink=10.0)

        output_file = tmp_path / "results.txt"

        with pytest.raises(ValueError, match="Unsupported format"):
            export_results(results, str(output_file), format="invalid")


# ============================================================================
# VISUALIZATION TESTS
# ============================================================================


class TestVisualization:
    """Test visualization functions."""

    def test_plot_mean_variance(self, large_data, tmp_path):
        """Test mean-variance plot."""
        M, design = large_data
        results = fit_differential(M, design, contrast=np.array([0, 1]), shrink=10.0)

        save_path = tmp_path / "mean_variance.png"

        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend

        fig = plot_mean_variance(results, save_path=str(save_path))

        assert save_path.exists()
        assert fig is not None

    def test_plot_mean_variance_no_save(self, large_data):
        """Test mean-variance plot without saving."""
        M, design = large_data
        results = fit_differential(M, design, contrast=np.array([0, 1]), shrink=10.0)

        import matplotlib

        matplotlib.use("Agg")

        fig = plot_mean_variance(results)
        assert fig is not None

    def test_plot_residual_diagnostics(self, simple_data, tmp_path):
        """Test residual diagnostics plot."""
        M, _, design = simple_data

        results, residuals = fit_differential(
            M, design, contrast=np.array([0, 1]), shrink="none", return_residuals=True
        )

        save_path = tmp_path / "residuals.png"

        import matplotlib

        matplotlib.use("Agg")

        fig = plot_residual_diagnostics(
            residuals, M, design, top_n=6, save_path=str(save_path)
        )

        assert save_path.exists()
        assert fig is not None

    def test_plot_sample_qc(self, data_with_missing, tmp_path):
        """Test sample QC plot."""
        M, metadata = data_with_missing

        save_path = tmp_path / "sample_qc.png"

        import matplotlib

        matplotlib.use("Agg")

        fig = plot_sample_qc(M, metadata, group_col="Type", save_path=str(save_path))

        assert save_path.exists()
        assert fig is not None

    def test_plot_sample_qc_few_complete_cpgs(self):
        """Test sample QC with few complete CpGs (triggers imputation)."""
        # Create data with lots of missing values
        M = pd.DataFrame(np.random.randn(50, 8))
        M.iloc[:45, :] = np.nan  # Only 5 complete CpGs

        metadata = pd.DataFrame(
            {"Type": ["Normal"] * 4 + ["Tumor"] * 4}, index=M.columns
        )

        import matplotlib

        matplotlib.use("Agg")

        fig = plot_sample_qc(M, metadata, group_col="Type")
        assert fig is not None

    def test_plot_pvalue_qq(self, large_data, tmp_path):
        """Test p-value Q-Q plot."""
        M, design = large_data
        results = fit_differential(M, design, contrast=np.array([0, 1]), shrink=10.0)

        save_path = tmp_path / "pvalue_qq.png"

        import matplotlib

        matplotlib.use("Agg")

        fig = plot_pvalue_qq(results, save_path=str(save_path))

        assert save_path.exists()
        assert fig is not None

    def test_plot_volcano_enhanced(self, large_data, tmp_path):
        """Test enhanced volcano plot."""
        M, design = large_data
        results = fit_differential(M, design, contrast=np.array([0, 1]), shrink=10.0)

        save_path = tmp_path / "volcano.png"

        import matplotlib

        matplotlib.use("Agg")

        fig = plot_volcano_enhanced(
            results, lfc_thresh=1.0, pval_thresh=0.05, top_n=5, save_path=str(save_path)
        )

        assert save_path.exists()
        assert fig is not None

    def test_plot_volcano_no_top_n(self, large_data):
        """Test volcano plot without labeling top hits."""
        M, design = large_data
        results = fit_differential(M, design, contrast=np.array([0, 1]), shrink=10.0)

        import matplotlib

        matplotlib.use("Agg")

        fig = plot_volcano_enhanced(results, top_n=0)
        assert fig is not None


# ============================================================================
# PDF LOGGER TESTS
# ============================================================================


class TestPDFLogger:
    """Test PDF logging functionality."""

    def test_pdflogger_init(self, tmp_path):
        """Test PDF logger initialization."""
        pdf_path = tmp_path / "test.pdf"
        logger = PDFLogger(str(pdf_path), echo=False)

        assert logger.path == str(pdf_path)
        assert not logger.echo
        assert len(logger.story) == 0

    def test_pdflogger_log_text_header1(self, tmp_path):
        """Test logging H1 header."""
        pdf_path = tmp_path / "test.pdf"
        logger = PDFLogger(str(pdf_path), echo=False)

        logger.log_text("# Main Title")
        assert len(logger.story) > 0

    def test_pdflogger_log_text_header2(self, tmp_path):
        """Test logging H2 header."""
        pdf_path = tmp_path / "test.pdf"
        logger = PDFLogger(str(pdf_path), echo=False)

        logger.log_text("## Section Title")
        assert len(logger.story) > 0

    def test_pdflogger_log_text_header3(self, tmp_path):
        """Test logging H3 header."""
        pdf_path = tmp_path / "test.pdf"
        logger = PDFLogger(str(pdf_path), echo=False)

        logger.log_text("### Subsection")
        assert len(logger.story) > 0

    def test_pdflogger_log_text_bullet(self, tmp_path):
        """Test logging bullet points."""
        pdf_path = tmp_path / "test.pdf"
        logger = PDFLogger(str(pdf_path), echo=False)

        logger.log_text("- Item 1")
        logger.log_text("- Item 2")

        assert logger.current_list is not None
        assert len(logger.current_list._flowables) == 2

    def test_pdflogger_log_text_paragraph(self, tmp_path):
        """Test logging plain text."""
        pdf_path = tmp_path / "test.pdf"
        logger = PDFLogger(str(pdf_path), echo=False)

        logger.log_text("Plain paragraph text")
        assert len(logger.story) > 0

    def test_pdflogger_log_text_empty(self, tmp_path):
        """Test logging empty string."""
        pdf_path = tmp_path / "test.pdf"
        logger = PDFLogger(str(pdf_path), echo=False)

        initial_len = len(logger.story)
        logger.log_text("")

        # Should not add anything
        assert len(logger.story) == initial_len

    def test_pdflogger_format_md_bold(self, tmp_path):
        """Test markdown bold formatting."""
        pdf_path = tmp_path / "test.pdf"
        logger = PDFLogger(str(pdf_path), echo=False)

        formatted = logger._format_md("**bold text**")
        assert "<b>bold text</b>" in formatted

    def test_pdflogger_format_md_italic(self, tmp_path):
        """Test markdown italic formatting."""
        pdf_path = tmp_path / "test.pdf"
        logger = PDFLogger(str(pdf_path), echo=False)

        formatted = logger._format_md("*italic text*")
        assert "<i>italic text</i>" in formatted

    def test_pdflogger_format_md_code(self, tmp_path):
        """Test markdown inline code formatting."""
        pdf_path = tmp_path / "test.pdf"
        logger = PDFLogger(str(pdf_path), echo=False)

        formatted = logger._format_md("`code text`")
        assert "Courier" in formatted

    def test_pdflogger_log_code(self, tmp_path):
        """Test logging code block."""
        pdf_path = tmp_path / "test.pdf"
        logger = PDFLogger(str(pdf_path), echo=False)

        logger.log_code("def function():\n    return True")
        assert len(logger.story) > 0

    def test_pdflogger_log_dataframe(self, tmp_path, simple_data):
        """Test logging dataframe."""
        M, _, _ = simple_data
        pdf_path = tmp_path / "test.pdf"
        logger = PDFLogger(str(pdf_path), echo=False)

        logger.log_dataframe(M.head(), title="Test Data", max_rows=5)
        assert len(logger.story) > 0

    def test_pdflogger_log_dataframe_large(self, tmp_path, large_data):
        """Test logging large dataframe (triggers truncation)."""
        M, _ = large_data
        pdf_path = tmp_path / "test.pdf"
        logger = PDFLogger(str(pdf_path), echo=False)

        logger.log_dataframe(M, max_rows=10)
        assert len(logger.story) > 0

    def test_pdflogger_log_image_exists(self, tmp_path):
        """Test logging existing image."""
        pdf_path = tmp_path / "test.pdf"
        logger = PDFLogger(str(pdf_path), echo=False)

        # Create a simple image
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        img_path = tmp_path / "test_img.png"
        fig.savefig(img_path)
        plt.close()

        logger.log_image(str(img_path), title="Test", caption="Caption")
        assert len(logger.story) > 0

    def test_pdflogger_log_image_missing(self, tmp_path):
        """Test logging non-existent image."""
        pdf_path = tmp_path / "test.pdf"
        logger = PDFLogger(str(pdf_path), echo=False)

        logger.log_image("/fake/path.png")
        # Should log missing image message
        assert len(logger.story) > 0

    def test_pdflogger_flush_list(self, tmp_path):
        """Test flushing list."""
        pdf_path = tmp_path / "test.pdf"
        logger = PDFLogger(str(pdf_path), echo=False)

        logger.log_text("- Item 1")
        assert logger.current_list is not None

        logger._flush_list()
        assert logger.current_list is None

    def test_pdflogger_echo(self, tmp_path, capsys):
        """Test echo functionality."""
        pdf_path = tmp_path / "test.pdf"
        logger = PDFLogger(str(pdf_path), echo=True)

        logger.log_text("Test message")

        captured = capsys.readouterr()
        assert "Test message" in captured.out

    def test_pdflogger_save(self, tmp_path):
        """Test saving PDF."""
        pdf_path = tmp_path / "test.pdf"
        logger = PDFLogger(str(pdf_path), echo=False)

        logger.log_text("# Test Report")
        logger.log_text("Some content")
        logger.save()

        assert pdf_path.exists()
        assert pdf_path.stat().st_size > 0

    def test_pdflogger_save_failure(self, tmp_path):
        """Test save failure handling."""
        # Create invalid path
        pdf_path = "/invalid/path/test.pdf"
        logger = PDFLogger(pdf_path, echo=False)

        logger.log_text("Test")

        # Should handle error gracefully
        logger.save()


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestIntegration:
    """End-to-end integration tests."""

    def test_full_pipeline(self):
        """Test complete analysis pipeline."""
        np.random.seed(1500)

        # 1. Simulate data
        n_cpg, n_samples = 1000, 12
        M = pd.DataFrame(
            np.random.randn(n_cpg, n_samples),
            index=[f"cg{i:08d}" for i in range(n_cpg)],
            columns=[f"S{i}" for i in range(n_samples)],
        )

        # Add differential methylation
        M.iloc[0:50, 6:] += 2.0

        # Add missing data
        missing_mask = np.random.random(M.shape) < 0.05
        M[missing_mask] = np.nan

        metadata = pd.DataFrame(
            {"Type": ["Normal"] * 6 + ["Tumor"] * 6}, index=M.columns
        )

        # 2. Preprocess
        M_filtered, _, _ = filter_cpgs_by_missingness(
            M, max_missing_rate=0.2, min_samples_per_group=4, groups=metadata["Type"]
        )

        M_imputed = impute_missing_values_fast(M_filtered, method="knn", k=3)

        M_ready = filter_min_per_group(
            M_imputed, groups=metadata["Type"], min_per_group=4, verbose=False
        )

        # 3. Analyze
        design = pd.DataFrame(
            {"Intercept": 1, "Tumor": (metadata["Type"] == "Tumor").astype(int)},
            index=M.columns,
        )

        results = fit_differential(
            M_ready, design, contrast=np.array([0, 1]), shrink=10.0, robust=True
        )

        # 4. Extract results
        sig_cpgs = get_significant_cpgs(results, lfc_thresh=1.0, pval_thresh=0.05)

        summary = summarize_differential_results(results, pval_thresh=0.05)

        # 5. Verify
        assert len(results) > 0
        assert summary["significant"] > 0
        assert len(sig_cpgs) == summary["significant"]

        # Should detect most of the 50 truly differential CpGs
        true_dm_cpgs = [
            f"cg{i:08d}" for i in range(50) if f"cg{i:08d}" in results.index
        ]
        detected = len(set(sig_cpgs) & set(true_dm_cpgs))
        sensitivity = detected / len(true_dm_cpgs) * 100

        assert sensitivity > 30  # Should detect at least 30%

    def test_pipeline_with_batch(self):
        """Test pipeline with batch effect adjustment."""
        np.random.seed(1500)

        # Simulate data with batch effect
        n_cpg, n_samples = 500, 12
        M = pd.DataFrame(
            np.random.randn(n_cpg, n_samples),
            index=[f"cg{i:08d}" for i in range(n_cpg)],
            columns=[f"S{i}" for i in range(n_samples)],
        )

        # Add differential methylation
        M.iloc[0:25, 6:] += 2.0

        # Add batch effect
        M.iloc[:, [1, 3, 5, 7, 9, 11]] -= 0.5  # Batch 1

        # Create design with batch
        design = pd.DataFrame(
            {
                "Intercept": 1,
                "Tumor": [0] * 6 + [1] * 6,
                "Batch": [0, 1, 0, 1, 0, 1] * 2,
            },
            index=M.columns,
        )

        # Test tumor effect adjusted for batch
        results = fit_differential(
            M,
            design,
            contrast=np.array([0, 1, 0]),  # Test tumor, adjust for batch
            shrink=10.0,
        )

        # Should still detect differential methylation
        sig_cpgs = get_significant_cpgs(results, lfc_thresh=1.0, pval_thresh=0.05)
        assert len(sig_cpgs) > 0

    def test_pipeline_all_outputs(self, tmp_path):
        """Test pipeline generates all expected outputs."""
        np.random.seed(1500)

        # Create data
        M = pd.DataFrame(
            np.random.randn(200, 10),
            index=[f"cg{i:08d}" for i in range(200)],
            columns=[f"S{i}" for i in range(10)],
        )
        M.iloc[0:10, 5:] += 2.0  # Add DM

        design = pd.DataFrame(
            {"Intercept": 1, "Group": [0] * 5 + [1] * 5}, index=M.columns
        )

        # Run analysis
        results = fit_differential(M, design, contrast=np.array([0, 1]), shrink=10.0)

        # Generate all outputs
        import matplotlib

        matplotlib.use("Agg")

        # Plots
        plot_mean_variance(results, save_path=str(tmp_path / "mean_var.png"))
        plot_pvalue_qq(results, save_path=str(tmp_path / "qq.png"))
        plot_volcano_enhanced(results, save_path=str(tmp_path / "volcano.png"))

        # Results
        export_results(results, str(tmp_path / "results.csv"), format="csv")

        # Summary
        summary = summarize_differential_results(results)

        # PDF report
        pdf = PDFLogger(str(tmp_path / "report.pdf"), echo=False)
        pdf.log_text("# Test Report")
        pdf.log_text(f"Significant CpGs: {summary['significant']}")
        pdf.log_image(str(tmp_path / "volcano.png"), "Volcano", "Volcano plot")
        pdf.save()

        # Verify all files exist
        assert (tmp_path / "mean_var.png").exists()
        assert (tmp_path / "qq.png").exists()
        assert (tmp_path / "volcano.png").exists()
        assert (tmp_path / "results.csv").exists()
        assert (tmp_path / "report.pdf").exists()


# ============================================================================
# EDGE CASES AND ERROR CONDITIONS
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_cpg(self):
        """Test analysis with single CpG."""
        M = pd.DataFrame(
            np.random.randn(1, 10),
            index=["cg00000000"],
            columns=[f"S{i}" for i in range(10)],
        )
        design = pd.DataFrame(
            {"Intercept": 1, "Group": [0] * 5 + [1] * 5}, index=M.columns
        )

        results = fit_differential(M, design, contrast=np.array([0, 1]), shrink="none")

        assert len(results) == 1
        assert "logFC" in results.columns

    def test_perfect_separation(self):
        """Test with perfect group separation."""
        M = pd.DataFrame(
            {
                **{f"S{i}": [0, 0, 0] for i in range(5)},
                **{f"S{i}": [5, 5, 5] for i in range(5, 10)},
            },
            index=["cg1", "cg2", "cg3"],
        )

        design = pd.DataFrame(
            {"Intercept": 1, "Group": [0] * 5 + [1] * 5}, index=M.columns
        )

        results = fit_differential(M, design, contrast=np.array([0, 1]), shrink="none")

        # Should detect perfect separation
        assert results["pval"].min() < 0.001

    def test_all_missing_cpg(self):
        """Test with CpG that has all missing values."""
        M = pd.DataFrame(np.random.randn(5, 6))
        M.iloc[2, :] = np.nan  # One completely missing CpG

        design = pd.DataFrame(
            {"Intercept": 1, "Group": [0] * 3 + [1] * 3}, index=M.columns
        )

        results = fit_differential(M, design, contrast=np.array([0, 1]), min_count=2)

        # Should exclude the all-missing CpG
        assert len(results) < len(M)

    def test_constant_cpg(self):
        """Test with constant CpG (no variance)."""
        M = pd.DataFrame(np.random.randn(5, 6))
        M.iloc[2, :] = 1.0  # Constant values

        design = pd.DataFrame(
            {"Intercept": 1, "Group": [0] * 3 + [1] * 3}, index=M.columns
        )

        with pytest.warns(UserWarning):
            results = fit_differential(
                M, design, contrast=np.array([0, 1]), shrink="none"
            )

        # Should handle gracefully
        assert len(results) <= len(M)

    def test_extreme_values(self):
        """Test with extreme M-values."""
        M = pd.DataFrame(
            np.random.randn(50, 10) * 10,  # Large variance
            columns=[f"S{i}" for i in range(10)],
        )
        design = pd.DataFrame(
            {"Intercept": 1, "Group": [0] * 5 + [1] * 5}, index=M.columns
        )

        # Should handle without error
        results = fit_differential(M, design, contrast=np.array([0, 1]), shrink=10.0)

        assert len(results) > 0
        assert np.isfinite(results["pval"]).all()


if __name__ == "__main__":
    pytest.main(
        [__file__, "-v", "--cov=core.engine", "--cov-report=html", "--cov-report=term"]
    )
