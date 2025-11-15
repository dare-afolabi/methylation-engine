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
    get_config,
    load_config,
    export_default_config,
    update_regional_pricing,
    get_platform_by_budget,
    PlannerConfig,
)

# Study Planning
from methylation_engine.core.planner import (
    # Core planning
    plan_sample_size,
    estimate_costs,
    estimate_timeline,
    plan_batches,
    assess_study_risks,
    create_study_plan,
    # Power analysis
    calculate_power,
    sample_size_for_power,
    detectable_effect_size,
    delta_beta_to_delta_m,
    # Comparisons
    compare_designs,
    compare_platforms,
    quick_recommendation,
    # Visualization
    plot_power_curves,
    plot_sample_size_heatmap,
)

# Differential Analysis
from methylation_engine.core.engine import (
    # Preprocessing
    filter_cpgs_by_missingness,
    impute_missing_values_fast,
    filter_min_per_group,
    # Analysis
    fit_differential,
    fit_differential_chunked,
    # Results
    get_significant_cpgs,
    summarize_differential_results,
    export_results,
    # Visualization
    plot_volcano_enhanced,
    plot_mean_variance,
    plot_pvalue_qq,
    plot_residual_diagnostics,
    plot_sample_qc,
    # Reporting
    PDFLogger,
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
    "plot_sample_size_heatmap",
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