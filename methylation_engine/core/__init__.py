#!/usr/bin/env python
# coding: utf-8

"""
Differential Methylation Analysis Pipeline

Toolkit for DNA methylation study planning and analysis.

Modules
-------
config : Configuration database for platforms, designs, costs
planner : Study planning (sample size, costs, timelines)
engine : Statistical analysis (empirical Bayes, differential methylation)
"""

__version__ = "0.2.0"
__author__ = "Dare Afolabi"
__email__ = "dare.afolabi@outlook.com"

# Configuration
from methylation_engine.core.config import (
    PlannerConfig,
    export_default_config,
    get_config,
    get_platform_by_budget,
    load_config,
    update_regional_pricing,
)

# Differential Analysis
from methylation_engine.core.engine import (
    PDFLogger,
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
)

# Study Planning
from methylation_engine.core.planner import (
    assess_study_risks,
    calculate_power,
    compare_designs,
    compare_platforms,
    create_study_plan,
    delta_beta_to_delta_m,
    detectable_effect_size,
    estimate_costs,
    estimate_timeline,
    plan_batches,
    plan_sample_size,
    plot_power_curves,
    quick_recommendation,
    sample_size_for_power,
)

__all__ = [
    # Version
    "__version__",
    # Config
    "get_config",
    "load_config",
    "export_default_config",
    "update_regional_pricing",
    "get_platform_by_budget",
    "PlannerConfig",
    # Planning
    "plan_sample_size",
    "estimate_costs",
    "estimate_timeline",
    "plan_batches",
    "assess_study_risks",
    "create_study_plan",
    "calculate_power",
    "sample_size_for_power",
    "detectable_effect_size",
    "delta_beta_to_delta_m",
    "compare_designs",
    "compare_platforms",
    "quick_recommendation",
    "plot_power_curves",
    # Engine
    "filter_cpgs_by_missingness",
    "impute_missing_values_fast",
    "filter_min_per_group",
    "fit_differential",
    "fit_differential_chunked",
    "get_significant_cpgs",
    "summarize_differential_results",
    "export_results",
    "plot_volcano_enhanced",
    "plot_mean_variance",
    "plot_pvalue_qq",
    "plot_residual_diagnostics",
    "plot_sample_qc",
    "PDFLogger",
]
