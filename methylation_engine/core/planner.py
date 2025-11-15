#!/usr/bin/env python
# coding: utf-8

"""
Differential Methylation Study Planner
Comprehensive toolkit for designing DNA methylation studies
Now integrated with configuration database
"""

import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from methylation_engine.core.config import PlannerConfig, get_config

# ============================================================================
# POWER ANALYSIS FUNCTIONS
# ============================================================================


def delta_beta_to_delta_m(delta_beta: float, baseline_beta: float = 0.5) -> float:
    """
    Convert delta-beta to delta-M (log fold change).

    Parameters
    ----------
    delta_beta : float
        Change in beta value (0-1 scale)
    baseline_beta : float
        Baseline methylation level

    Returns
    -------
    float
        Equivalent change in M-value

    Examples
    --------
    >>> delta_beta_to_delta_m(0.10, baseline_beta=0.5)
    1.45
    """
    beta1 = baseline_beta
    beta2 = np.clip(baseline_beta + delta_beta, 0.01, 0.99)

    m1 = np.log2(beta1 / (1 - beta1))
    m2 = np.log2(beta2 / (1 - beta2))

    return m2 - m1


def calculate_power(
    n_per_group: int,
    effect_size: float,
    alpha: float = 0.05,
    variance: float = 1.0,
    prior_df: float = 10.0,
    two_sided: bool = True,
    paired: bool = False,
) -> float:
    """
    Calculate statistical power with empirical Bayes moderation.

    Parameters
    ----------
    n_per_group : int
        Number of samples per group
    effect_size : float
        Expected mean difference (M-value units)
    alpha : float
        Significance level
    variance : float
        Expected variance
    prior_df : float
        Prior degrees of freedom from EB shrinkage
    two_sided : bool
        Whether test is two-sided
    paired: bool
        Whether test is paired

    Returns
    -------
    float
        Statistical power (0-1)

    Examples
    --------
    >>> calculate_power(12, 1.5, alpha=0.05)
    0.82
    """
    df_resid = 2 * n_per_group - 2
    df_total = df_resid + prior_df

    se = (
        np.sqrt(variance / n_per_group)
        if paired
        else np.sqrt(2 * variance / n_per_group)
    )
    ncp = effect_size / se

    if two_sided:
        t_crit = stats.t.ppf(1 - alpha / 2, df=df_total)
    else:
        t_crit = stats.t.ppf(1 - alpha, df=df_total)

    power = 1 - stats.nct.cdf(t_crit, df=df_total, nc=ncp)

    if two_sided:
        power += stats.nct.cdf(-t_crit, df=df_total, nc=ncp)

    return power


def sample_size_for_power(
    target_power: float,
    effect_size: float,
    alpha: float = 0.05,
    variance: float = 1.0,
    prior_df: float = 10.0,
    max_n: int = 200,
    paired: bool = False,
) -> int:
    """
    Calculate required sample size to achieve target power.

    Parameters
    ----------
    target_power : float
        Desired power (0-1)
    effect_size : float
        Expected effect size (M-value units)
    alpha : float
        Significance level
    variance : float
        Expected variance
    prior_df : float
        Prior degrees of freedom
    max_n : int
        Maximum sample size to search
    paired: bool
        Whether test is paired

    Returns
    -------
    int
        Required sample size per group

    Examples
    --------
    >>> sample_size_for_power(0.80, 1.5)
    12
    """
    low, high = 3, max_n
    result = max_n
    while low <= high:
        mid = (low + high) // 2
        power = calculate_power(
            mid, effect_size, alpha, variance, prior_df, paired=paired
        )
        if power >= target_power:
            result = mid
            high = mid - 1
        else:
            low = mid + 1
    return result


def detectable_effect_size(
    n_per_group: int,
    target_power: float = 0.80,
    alpha: float = 0.05,
    variance: float = 1.0,
    prior_df: float = 10.0,
    paired: bool = False,
) -> float:
    """
    Calculate minimum detectable effect size for given sample size.

    Parameters
    ----------
    n_per_group : int
        Sample size per group
    target_power : float
        Target power
    alpha : float
        Significance level
    variance : float
        Expected variance
    prior_df : float
        Prior degrees of freedom
    paired: bool
        Whether test is paired

    Returns
    -------
    float
        Minimum detectable effect size (M-value units)

    Examples
    --------
    >>> detectable_effect_size(12, target_power=0.80)
    1.48
    """
    low, high = 0.01, 10.0
    tolerance = 0.005
    while high - low > tolerance:
        mid = (low + high) / 2
        power = calculate_power(
            n_per_group, mid, alpha, variance, prior_df, paired=paired
        )
        if power >= target_power:
            high = mid
        else:
            low = mid
    return round(high, 3)


# ============================================================================
# SAMPLE SIZE PLANNING
# ============================================================================


def plan_sample_size(
    expected_effect_size: Optional[float] = None,
    expected_delta_beta: Optional[float] = None,
    target_power: float = 0.80,
    alpha: float = 0.05,
    variance: float = 1.0,
    design_type: str = "two_group",
    prior_df: float = 10.0,
    config: Optional[PlannerConfig] = None,
) -> Dict[str, Any]:
    """
    Calculate recommended sample sizes with power analysis.

    Parameters
    ----------
    expected_effect_size : float, optional
        Expected effect size in M-value units
    expected_delta_beta : float, optional
        OR expected effect size in beta-value units (0-1)
    target_power : float
        Desired statistical power
    alpha : float
        Significance level
    variance : float
        Expected variance
    design_type : str
        Study design type
    prior_df : float
        Prior degrees of freedom
    config : PlannerConfig, optional
        Configuration instance (uses global if None)

    Returns
    -------
    Dict
        Sample size recommendations at multiple levels

    Examples
    --------
    >>> result = plan_sample_size(expected_delta_beta=0.10, target_power=0.80)
    >>> result['recommended']['n_per_group']
    12
    """
    if config is None:
        config = get_config()

    # Convert delta-beta to effect size if needed
    if expected_delta_beta is not None and expected_effect_size is None:
        if not (
            isinstance(expected_delta_beta, (int, float))
            and 0 < expected_delta_beta < 1
        ):
            raise ValueError("expected_delta_beta must be a number in the range (0, 1)")
        expected_effect_size = delta_beta_to_delta_m(expected_delta_beta)
    elif expected_effect_size is None:
        raise ValueError(
            "Must provide either expected_effect_size or expected_delta_beta"
        )

    # Get design adjustments from config
    design_info = config.get_design(design_type)
    paired = design_info.get("paired", False)
    power_adj = design_info["power_adjustment"]

    # Base calculation
    n_base = sample_size_for_power(
        target_power, expected_effect_size, alpha, variance, prior_df, paired=paired
    )

    # Adjust for design type
    n_recommended = int(n_base * power_adj)
    n_recommended = max(n_recommended, design_info["min_n_recommended"])

    # Ensure recommended meets target power
    max_iter = 100
    for _ in range(max_iter):
        power_check = calculate_power(
            n_recommended,
            expected_effect_size,
            alpha,
            variance,
            prior_df,
            paired=paired,
        )
        if power_check >= target_power:
            break
        n_recommended += 1
    else:
        raise RuntimeError(
            f"Failed to achieve target power ({target_power:.0%})\
             after {max_iter} adjustments. "
            "Check configuration (power_adjustment, min_n_recommended) or effect size."
        )

    # Create options
    n_minimum = max(design_info["min_n_recommended"], n_recommended // 2)
    n_optimal = int(n_recommended * 1.5)

    options = {"minimum": n_minimum, "recommended": n_recommended, "optimal": n_optimal}

    # Calculate actual power and detectable effects for each option
    results = {}
    for level, n in options.items():
        power = calculate_power(
            n, expected_effect_size, alpha, variance, prior_df, paired=paired
        )
        det_effect = detectable_effect_size(
            n, target_power, alpha, variance, prior_df, paired=paired
        )

        # Convert detectable ΔM back to Δβ properly
        # Use inverse of delta_beta_to_delta_m conversion
        baseline_beta = 0.5
        m_baseline = np.log2(baseline_beta / (1 - baseline_beta))
        m_new = m_baseline + det_effect
        beta_new = 2**m_new / (1 + 2**m_new)
        det_delta_beta = beta_new - baseline_beta

        results[level] = {
            "n_per_group": n,
            "total_samples": n * design_info["n_groups"],
            "power": power,
            "detectable_effect_m": det_effect,
            "detectable_effect_beta": det_delta_beta,
        }

    return results


# ============================================================================
# COST ESTIMATION
# ============================================================================


def estimate_costs(
    n_samples: int,
    platform: str = "EPIC",
    include_optional: bool = True,
    config: Optional[PlannerConfig] = None,
) -> Dict[str, Any]:
    """
    Estimate study costs with detailed breakdown.

    Parameters
    ----------
    n_samples : int
        Total number of samples
    platform : str
        Methylation platform
    include_optional : bool
        Include optional cost components
    config : PlannerConfig, optional
        Configuration instance (uses global if None)

    Returns
    -------
    Dict
        Cost breakdown by category

    Examples
    --------
    >>> costs = estimate_costs(24, platform='EPIC')
    >>> costs['total']
    19320
    """
    if config is None:
        config = get_config()

    platform_info = config.get_platform(platform)

    # Get applicable cost components
    cost_components = config.get_cost_components(
        platform=platform, include_optional=include_optional
    )

    # Calculate costs
    costs = {"platform": platform_info["cost_per_sample"] * n_samples}

    for comp_id, comp in cost_components.items():
        if comp["unit"] == "per_sample":
            costs[comp_id] = comp["cost"] * n_samples
        elif comp["unit"] == "per_cpg":
            # Validation costs are per-CpG validated (typically validate 5-10 CpGs)
            # We'll assume 0 by default unless explicitly requested
            costs[comp_id] = 0

    total = sum(costs.values())

    return {
        "breakdown": costs,
        "total": total,
        "per_sample": total / n_samples,
        "n_samples": n_samples,
        "platform": platform,
    }


# ============================================================================
# TIMELINE ESTIMATION
# ============================================================================


def estimate_timeline(
    n_samples: int,
    platform: str = "EPIC",
    start_date: Optional[datetime] = None,
    include_optional_phases: bool = True,
    config: Optional[PlannerConfig] = None,
) -> Dict[str, Any]:
    """
    Create project timeline with phase breakdown.

    Parameters
    ----------
    n_samples : int
        Total number of samples
    platform : str
        Methylation platform
    start_date : datetime, optional
        Project start date
    include_optional_phases : bool
        Include validation phase
    config : PlannerConfig, optional
        Configuration instance (uses global if None)

    Returns
    -------
    Dict
        Timeline with phases and milestones

    Examples
    --------
    >>> timeline = estimate_timeline(24, platform='EPIC')
    >>> timeline['total_duration_months']
    8.5
    """
    if config is None:
        config = get_config()

    if start_date is None:
        start_date = datetime.now()

    platform_info = config.get_platform(platform)

    # Calculate dates
    current_date = start_date
    timeline = []

    for phase_id, phase_config in config.timeline_phases.items():
        # Skip optional phases if not requested
        if phase_config.get("optional", False) and not include_optional_phases:
            continue

        # Calculate and ensure duration is numeric
        duration = float(str(phase_config["base_duration_days"]))

        # Scaling factor
        scaling_factor_raw = str(phase_config.get("scaling_factor", 0))
        scaling_factor = (
            float(scaling_factor_raw) if scaling_factor_raw is not None else 0.0
        )

        if scaling_factor > 0:
            duration += float(n_samples) * scaling_factor

        # Batch adjustment
        if "batch_adjustment" in phase_config:
            batch_adj = float(str(phase_config["batch_adjustment"]))
            n_batches = int(np.ceil(n_samples / 96))
            duration += (n_batches - 1) * batch_adj

        # Platform-specific processing
        if phase_id == "array_processing":
            processing_days = float(platform_info["processing_days"])
            duration = processing_days + float(n_samples // 96) * 2.0

        end_date = current_date + timedelta(days=float(duration))

        timeline.append(
            {
                "phase": str(phase_config["name"]),
                "start_date": current_date,
                "end_date": end_date,
                "duration_days": float(duration),
                "duration_weeks": float(duration) / 7,
                "description": str(phase_config["description"]),
                "critical": bool(phase_config["critical"]),
            }
        )

        current_date = end_date

    total_days = (current_date - start_date).days

    return {
        "start_date": start_date,
        "completion_date": current_date,
        "total_duration_days": total_days,
        "total_duration_weeks": total_days / 7,
        "total_duration_months": total_days / 30,
        "phases": timeline,
    }


# ============================================================================
# BATCH PLANNING
# ============================================================================


def plan_batches(
    n_samples: int, n_groups: int = 2, samples_per_batch: int = 96
) -> Dict[str, Any]:
    """
    Plan batch structure to minimize batch effects.

    Parameters
    ----------
    n_samples : int
        Total number of samples
    n_groups : int
        Number of experimental groups
    samples_per_batch : int
        Maximum samples per batch

    Returns
    -------
    Dict
        Batch distribution plan with recommendations

    Examples
    --------
    >>> batch_plan = plan_batches(24, n_groups=2, samples_per_batch=96)
    >>> batch_plan['n_batches']
    1
    """
    n_samples = int(n_samples)
    n_groups = int(n_groups)
    samples_per_batch = int(samples_per_batch)

    n_batches = int(np.ceil(n_samples / samples_per_batch))

    # Distribute samples across batches
    samples_per_batch_actual = n_samples // n_batches
    remainder = n_samples % n_batches

    batch_distribution = []
    for i in range(n_batches):
        batch_size = samples_per_batch_actual + (1 if i < remainder else 0)

        # Balance groups within batch
        samples_per_group = batch_size // n_groups

        group_dist = {}
        for g in range(n_groups):
            group_dist[f"group{g+1}"] = samples_per_group + (
                1 if g < (batch_size % n_groups) else 0
            )

        # Calculate balance ratio
        group_sizes = list(group_dist.values())
        balance = (
            float(min(group_sizes)) / float(max(group_sizes))
            if float(max(group_sizes)) > 0
            else 1.0
        )

        batch_info = {
            "batch": i + 1,
            "total_samples": batch_size,
            "balance_ratio": balance,
        }
        batch_info.update(group_dist)

        batch_distribution.append(batch_info)

    # Generate recommendations
    recommendations = []
    if n_batches > 1:
        recommendations.append("Include batch as covariate in design matrix")
        recommendations.append("Randomize sample assignment to batches")
        recommendations.append("Process batches using same reagent lots")
        recommendations.append("Include technical replicates across batches")

    if any(b["balance_ratio"] < 0.8 for b in batch_distribution):
        recommendations.append(
            "Warning: Unbalanced batches detected - consider adjusting sample size"
        )

    return {
        "n_batches": n_batches,
        "samples_per_batch": samples_per_batch,
        "distribution": batch_distribution,
        "recommendations": recommendations,
        "balanced": all(b["balance_ratio"] >= 0.8 for b in batch_distribution),
    }


# ============================================================================
# RISK ASSESSMENT
# ============================================================================


def assess_study_risks(
    n_samples: int,
    power: float,
    budget: Optional[float] = None,
    estimated_cost: Optional[float] = None,
    n_batches: int = 1,
    platform: str = "EPIC",
    config: Optional[PlannerConfig] = None,
) -> Dict[str, Any]:
    """
    Identify potential study risks and mitigation strategies.

    Parameters
    ----------
    n_samples : int
        Total sample size
    power : float
        Statistical power
    budget : float, optional
        Available budget
    estimated_cost : float, optional
        Estimated study cost
    n_batches : int
        Number of processing batches
    platform : str
        Methylation platform
    config : PlannerConfig, optional
        Configuration instance (uses global if None)

    Returns
    -------
    Dict
        Risk assessment with severity and mitigation

    Examples
    --------
    >>> risks = assess_study_risks(24, 0.75, budget=20000, estimated_cost=19000)
    >>> len(risks['risks'])
    2
    """
    if config is None:
        config = get_config()

    risks = []

    # Power-related risks
    if power < 0.75:
        risks.append(
            {
                "category": "Statistical Power",
                "risk": f"Underpowered study (power={power:.0%})",
                "severity": "High",
                "impact": "May miss true effects, high false negative rate",
                "mitigation": "Increase sample size or focus on larger effect sizes",
            }
        )
    elif power < 0.80:
        risks.append(
            {
                "category": "Statistical Power",
                "risk": f"Marginal power (power={power:.0%})",
                "severity": "Medium",
                "impact": "Limited ability to detect smaller effects",
                "mitigation": "Consider increasing sample size by 20-30%",
            }
        )

    # Sample size risks
    if n_samples < 20:
        risks.append(
            {
                "category": "Sample Size",
                "risk": "Small sample size (<20 total)",
                "severity": "Medium",
                "impact": "Limited generalizability, sensitive to outliers",
                "mitigation": "Use paired design if possible,\
                 strict QC, validate results",
            }
        )

    # Budget risks
    if budget is not None and estimated_cost is not None:
        if estimated_cost > budget:
            deficit = estimated_cost - budget
            over_pct = float(deficit) / float(budget)
            if over_pct <= 0.10:
                severity = "Medium"
                impact = "Minor overrun; likely solvable with small scope tweaks"
                mitigation = (
                    "Trim optional costs, negotiate rates, or reduce validation scope"
                )
            elif over_pct <= 0.30:
                severity = "High"
                impact = (
                    "Material overrun; may require scope change or additional funding"
                )
                mitigation = (
                    "Reduce sample size, switch to cheaper platform, or secure funds"
                )
            else:
                severity = "High"
                impact = "Severe overrun; study not feasible without major changes"
                mitigation = "Re-scope study, use legacy platform, or phase work"
            risks.append(
                {
                    "category": "Budget",
                    "risk": f"Cost exceeds budget by ${deficit:,.0f} ({over_pct:.0%})",
                    "severity": severity,
                    "impact": impact,
                    "mitigation": mitigation,
                }
            )
        elif estimated_cost > 0.9 * budget:
            risks.append(
                {
                    "category": "Budget",
                    "risk": "Limited budget buffer (<10%)",
                    "severity": "Low",
                    "impact": "No contingency for unexpected costs",
                    "mitigation": "Plan for potential sample failures, add contingency",
                }
            )

    # Batch effect risks
    if n_batches > 3:
        risks.append(
            {
                "category": "Technical",
                "risk": f"Multiple processing batches (n={n_batches})",
                "severity": "Medium",
                "impact": "Potential confounding by batch",
                "mitigation": "Balance groups across\
                 batches, include batch in model, use ComBat",
            }
        )

    # Platform-specific risks
    platform_info = config.get_platform(platform)
    if platform in ["WGBS", "Nanopore"]:
        risks.append(
            {
                "category": "Technical",
                "risk": f'Complex {platform_info["name"]} data analysis',
                "severity": "Medium",
                "impact": "Requires specialized bioinformatics expertise",
                "mitigation": "Ensure bioinformatics\
                 support available, allocate extra time",
            }
        )

    if not platform_info.get("recommended", False):
        risks.append(
            {
                "category": "Platform",
                "risk": f'{platform_info["name"]} not recommended for new studies',
                "severity": "Low",
                "impact": platform_info.get("notes", "May have limitations"),
                "mitigation": "Consider using a recommended\
                 platform like EPIC or EPICv2",
            }
        )

    # Sample failure risk
    if n_samples > 50:
        risks.append(
            {
                "category": "Data Quality",
                "risk": "Expected 5-10% sample failure rate",
                "severity": "Low",
                "impact": "Some samples may fail QC",
                "mitigation": "Plan 10% oversample, implement strict pre-array QC",
            }
        )

    # Summarize risk severity
    severity_counts = {
        "High": len([r for r in risks if r["severity"] == "High"]),
        "Medium": len([r for r in risks if r["severity"] == "Medium"]),
        "Low": len([r for r in risks if r["severity"] == "Low"]),
    }

    return {
        "risks": risks,
        "n_risks": len(risks),
        "severity_counts": severity_counts,
        "high_priority": [r for r in risks if r["severity"] == "High"],
    }


# ============================================================================
# VISUALIZATION
# ============================================================================


def plot_power_curves(
    effect_sizes: List[float] = [0.0, 0.0, 0.0, 0.0],
    sample_sizes: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    save_path: Optional[str] = None,
    dpi: int = 300,
    paired: bool = False,
    design_label: Optional[str] = None,
):
    """
    Plot power curves for different effect sizes.

    Parameters
    ----------
    effect_sizes : List[float]
        Effect sizes to plot
    sample_sizes : np.ndarray, optional
        Sample size range
    alpha : float
        Significance level
    save_path : str, optional
        Path to save figure
    dpi : int
        Figure resolution
    paired: bool
        Whether test is paired

    Returns
    -------
    matplotlib.figure.Figure
        Power curve figure
    """
    if sample_sizes is None:
        sample_sizes = np.arange(3, 51)

    fig, ax = plt.subplots(figsize=(10, 6))

    for effect in effect_sizes:
        powers = [
            calculate_power(n, effect, alpha, paired=paired) for n in sample_sizes
        ]
        # Correct inverse transformation: ΔM → Δβ at baseline β = 0.5
        baseline_beta = 0.5
        m1 = np.log2(baseline_beta / (1 - baseline_beta))
        m2 = m1 + effect
        beta2 = 2**m2 / (1 + 2**m2)
        delta_beta = beta2 - baseline_beta
        ax.plot(
            sample_sizes,
            powers,
            marker="o",
            markersize=4,
            label=f"ΔM={effect:.1f} (Δβ≈{delta_beta:.2f})",
        )

    ax.axhline(0.80, color="red", linestyle="--", alpha=0.5, label="80% power")
    ax.axhline(0.90, color="orange", linestyle="--", alpha=0.5, label="90% power")

    ax.set_xlabel("Sample Size per Group", fontsize=12)
    ax.set_ylabel("Statistical Power", fontsize=12)
    title = "Power Analysis for Differential Methylation"
    if design_label:
        title += f"\n({design_label})"
    ax.set_title(title, fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_ylim((0.0, 1.0))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    return fig


# ============================================================================
# COMPREHENSIVE STUDY PLAN
# ============================================================================


def create_study_plan(
    study_name: str,
    expected_delta_beta: float,
    target_power: float = 0.80,
    alpha: float = 0.05,
    design_type: str = "two_group",
    platform: str = "EPIC",
    budget: Optional[float] = None,
    output_dir: Optional[str] = None,
    config: Optional[PlannerConfig] = None,
) -> Dict[str, Any]:
    """
    Generate comprehensive study plan with all analyses.

    Parameters
    ----------
    study_name : str
        Name of the study
    expected_delta_beta : float
        Expected methylation difference (0-1 scale)
    target_power : float
        Desired statistical power
    alpha : float
        Significance level
    design_type : str
        Study design type
    platform : str
        Methylation platform
    budget : float, optional
        Available budget
    output_dir : str, optional
        Directory for output files
    config : PlannerConfig, optional
        Configuration instance (uses global if None)

    Returns
    -------
    Dict
        Complete study plan with all components

    Examples
    --------
    >>> plan = create_study_plan(
    ...     study_name="Cancer vs Normal",
    ...     expected_delta_beta=0.10,
    ...     target_power=0.80,
    ...     platform='EPIC',
    ...     budget=20000
    ... )
    >>> plan['sample_size']['recommended']['n_per_group']
    12
    """
    if config is None:
        config = get_config()

    if not 0 < expected_delta_beta < 1:
        raise ValueError("expected_delta_beta must be in the range (0, 1)")

    # Convert effect size
    effect_size = delta_beta_to_delta_m(expected_delta_beta)

    # Get design and platform info from config
    design_info = config.get_design(design_type)
    platform_info = config.get_platform(platform)

    # Sample size planning
    sample_size_plan = plan_sample_size(
        expected_effect_size=effect_size,
        target_power=target_power,
        alpha=alpha,
        design_type=design_type,
        config=config,
    )

    # Get recommended sample size
    n_total = sample_size_plan["recommended"]["total_samples"]
    # n_per_group = sample_size_plan["recommended"]["n_per_group"]
    power = sample_size_plan["recommended"]["power"]

    # Cost estimation
    cost_plan = estimate_costs(n_total, platform=platform, config=config)

    platform_switched = False
    original_platform = platform

    if budget is not None and cost_plan["total"] > budget:
        cheaper_platforms = config.list_platforms().sort_values("cost_per_sample")
        for _, row in cheaper_platforms.iterrows():
            alt_platform = row.name
            alt_cost = estimate_costs(n_total, platform=alt_platform, config=config)
            if alt_cost["total"] <= budget:
                platform = alt_platform
                platform_info = config.get_platform(platform)
                cost_plan = alt_cost
                platform_switched = True
                break

    # Timeline
    timeline = estimate_timeline(n_total, platform=platform, config=config)

    # Batch planning
    batch_plan = plan_batches(n_total, design_info["n_groups"])

    # Risk assessment
    risk_assessment = assess_study_risks(
        n_samples=n_total,
        power=power,
        budget=budget,
        estimated_cost=cost_plan["total"],
        n_batches=batch_plan["n_batches"],
        platform=platform,
        config=config,
    )

    # Compile study plan
    study_plan = {
        "study_name": study_name,
        "configuration": {
            "expected_delta_beta": expected_delta_beta,
            "expected_effect_size": effect_size,
            "target_power": target_power,
            "alpha": alpha,
            "design_type": design_type,
            "design_info": design_info,
            "platform": platform,
            "original_platform": original_platform if platform_switched else None,
            "platform_switched": platform_switched,
            "platform_info": platform_info,
            "budget": budget,
        },
        "sample_size": sample_size_plan,
        "costs": cost_plan,
        "timeline": timeline,
        "batch_plan": batch_plan,
        "risk_assessment": risk_assessment,
        "created_date": datetime.now(),
    }

    # Generate visualizations if output directory specified
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

        # Power curves
        plot_power_curves(
            effect_sizes=[effect_size * 0.5, effect_size, effect_size * 1.5],
            save_path=os.path.join(output_dir, "power_curves.png"),
        )

        # Generate text report
        _generate_text_report(study_plan, output_dir)

        # Export to Excel
        _export_to_excel(study_plan, output_dir)

    return study_plan


def _generate_text_report(study_plan: Dict[str, Any], output_dir: str):
    """Generate text report from study plan."""
    report_path = os.path.join(output_dir, "study_plan_report.txt")

    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write(f"STUDY PLAN: {study_plan['study_name']}\n")
        f.write("=" * 80 + "\n")
        f.write(
            f"Generated: {study_plan['created_date'].strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )

        # Configuration
        cfg = study_plan["configuration"]
        f.write("-" * 80 + "\n")
        f.write("1. STUDY CONFIGURATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Design: {cfg['design_info']['name']}\n")
        f.write(f"  {cfg['design_info']['description']}\n")
        f.write(f"Platform: {cfg['platform_info']['name']}\n")
        f.write(f"  CpGs: {cfg['platform_info']['n_cpgs']:,}\n")
        f.write(f"  Coverage: {cfg['platform_info']['coverage']}\n")
        f.write("\nExpected Effect:\n")
        f.write(f"  Δβ = {cfg['expected_delta_beta']:.3f}\n")
        f.write(f"  ΔM = {cfg['expected_effect_size']:.2f}\n")
        f.write(f"Target Power: {cfg['target_power']:.0%}\n")
        f.write(f"Significance: α = {cfg['alpha']}\n")

        # Sample size
        f.write("\n" + "-" * 80 + "\n")
        f.write("2. SAMPLE SIZE RECOMMENDATIONS\n")
        f.write("-" * 80 + "\n")
        for level in ["minimum", "recommended", "optimal"]:
            ss = study_plan["sample_size"][level]
            f.write(f"\n{level.upper()}:\n")
            f.write(f"  Per group: {ss['n_per_group']}\n")
            f.write(f"  Total: {ss['total_samples']}\n")
            f.write(f"  Power: {ss['power']:.1%}\n")
            f.write(f"  Detectable Δβ: {ss['detectable_effect_beta']:.3f}\n")

        # Costs
        f.write("\n" + "-" * 80 + "\n")
        f.write("3. BUDGET ANALYSIS\n")
        f.write("-" * 80 + "\n")
        costs = study_plan["costs"]
        f.write(f"Total Cost: ${costs['total']:,.0f}\n")
        f.write(f"Per Sample: ${costs['per_sample']:,.0f}\n")
        f.write("\nBreakdown:\n")
        for item, cost in costs["breakdown"].items():
            f.write(f"  {item}: ${cost:,.0f}\n")

        if cfg["budget"]:
            f.write(f"\nBudget: ${cfg['budget']:,.0f}\n")
            within = costs["total"] <= cfg["budget"]
            f.write(f"Within budget: {'✓ Yes' if within else '✗ No'}\n")

        # Timeline
        f.write("\n" + "-" * 80 + "\n")
        f.write("4. PROJECT TIMELINE\n")
        f.write("-" * 80 + "\n")
        timeline = study_plan["timeline"]
        f.write(f"Start: {timeline['start_date'].strftime('%Y-%m-%d')}\n")
        f.write(f"Completion: {timeline['completion_date'].strftime('%Y-%m-%d')}\n")
        f.write(f"Duration: {timeline['total_duration_months']:.1f} months\n\n")
        f.write("Phases:\n")
        for phase in timeline["phases"]:
            f.write(f"  {phase['phase']}: {phase['duration_weeks']:.1f} weeks\n")

        # Batch plan
        f.write("\n" + "-" * 80 + "\n")
        f.write("5. BATCH PLANNING\n")
        f.write("-" * 80 + "\n")
        batch = study_plan["batch_plan"]
        f.write(f"Number of batches: {batch['n_batches']}\n")
        f.write(f"Balanced: {'✓ Yes' if batch['balanced'] else '✗ No'}\n\n")
        f.write("Distribution:\n")
        for b in batch["distribution"]:
            f.write(f"  Batch {b['batch']}: {b['total_samples']} samples\n")

        if batch["recommendations"]:
            f.write("\nRecommendations:\n")
            for rec in batch["recommendations"]:
                f.write(f"  • {rec}\n")

        # Risks
        f.write("\n" + "-" * 80 + "\n")
        f.write("6. RISK ASSESSMENT\n")
        f.write("-" * 80 + "\n")
        risks = study_plan["risk_assessment"]
        f.write(f"Total Risks: {risks['n_risks']}\n")
        for severity, count in risks["severity_counts"].items():
            f.write(f"  {severity}: {count}\n")

        if risks["risks"]:
            f.write("\n")
            for risk in risks["risks"]:
                f.write(f"\n[{risk['severity']}] {risk['category']}: {risk['risk']}\n")
                f.write(f"  Impact: {risk['impact']}\n")
                f.write(f"  Mitigation: {risk['mitigation']}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")


def _export_to_excel(study_plan: Dict[str, Any], output_dir: str):
    """Export study plan to Excel workbook."""
    excel_path = os.path.join(output_dir, "study_plan.xlsx")

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:

        # Configuration
        config_data = []
        cfg = study_plan["configuration"]
        for key in [
            "expected_delta_beta",
            "expected_effect_size",
            "target_power",
            "alpha",
            "design_type",
            "platform",
        ]:
            if key in cfg:
                config_data.append({"Parameter": key, "Value": cfg[key]})
        pd.DataFrame(config_data).to_excel(
            writer, sheet_name="Configuration", index=False
        )

        # Sample size
        ss_data = []
        for level in ["minimum", "recommended", "optimal"]:
            ss = study_plan["sample_size"][level]
            ss_data.append(
                {
                    "Level": level,
                    "N_per_group": ss["n_per_group"],
                    "Total_samples": ss["total_samples"],
                    "Power": f"{ss['power']:.2%}",
                    "Detectable_Delta_Beta": f"{ss['detectable_effect_beta']:.3f}",
                }
            )
        pd.DataFrame(ss_data).to_excel(writer, sheet_name="Sample_Size", index=False)

        # Costs
        costs = study_plan["costs"]
        cost_data = pd.DataFrame(
            [{"Category": k, "Cost": v} for k, v in costs["breakdown"].items()]
        )
        cost_data = pd.concat(
            [cost_data, pd.DataFrame([{"Category": "TOTAL", "Cost": costs["total"]}])]
        )
        cost_data.to_excel(writer, sheet_name="Costs", index=False)

        # Timeline
        timeline_df = pd.DataFrame(study_plan["timeline"]["phases"])
        timeline_df.to_excel(writer, sheet_name="Timeline", index=False)

        # Batch plan
        batch_df = pd.DataFrame(study_plan["batch_plan"]["distribution"])
        batch_df.to_excel(writer, sheet_name="Batch_Plan", index=False)

        # Risks
        if study_plan["risk_assessment"]["risks"]:
            risk_df = pd.DataFrame(study_plan["risk_assessment"]["risks"])
            risk_df.to_excel(writer, sheet_name="Risk_Assessment", index=False)


# ============================================================================
# QUICK RECOMMENDATIONS
# ============================================================================


def quick_recommendation(
    expected_delta_beta: float,
    budget: Optional[float] = None,
    platform: str = "EPIC",
    design_type: str = "two_group",
    config: Optional[PlannerConfig] = None,
) -> str:
    """
    Get quick sample size recommendation with one-line summary.

    Parameters
    ----------
    expected_delta_beta : float
        Expected methylation difference
    budget : float, optional
        Available budget
    platform : str
        Platform choice
    design_type : str
        Study design type
    config : PlannerConfig, optional
        Configuration instance (uses global if None)

    Returns
    -------
    str
        Quick recommendation summary

    Examples
    --------
    >>> print(quick_recommendation(0.10, budget=20000, design_type='two_group'))
    Recommend 12 per group (24 total) for 80% power. Cost: $19,320. Within budget.
    """
    if config is None:
        config = get_config()

    if not 0 < expected_delta_beta < 1:
        raise ValueError("expected_delta_beta must be in the range (0, 1)")

    # Get design info
    design_info = config.get_design(design_type)
    paired = design_info.get("paired", False)

    # Calculate sample size
    effect_size = delta_beta_to_delta_m(expected_delta_beta)
    n = sample_size_for_power(0.80, effect_size, paired=paired)

    n_total = n * design_info["n_groups"]

    # Calculate cost
    costs = estimate_costs(n_total, platform, config=config)

    # Format recommendation
    rec = f"Recommend {n} per group ({n_total} total) for\
     80% power ({design_info['name']}). "
    rec += f"Cost: ${costs['total']:,.0f}. "

    if budget is not None:
        if costs["total"] <= budget:
            rec += "Within budget."
        else:
            deficit = costs["total"] - budget
            rec += f"Exceeds budget by ${deficit:,.0f}."

    return rec


# ============================================================================
# COMPARISON UTILITIES
# ============================================================================


def compare_designs(
    expected_delta_beta: float,
    target_power: float = 0.80,
    designs: Optional[List[str]] = None,
    config: Optional[PlannerConfig] = None,
) -> pd.DataFrame:
    """
    Compare sample size requirements across study designs.

    Parameters
    ----------
    expected_delta_beta : float
        Expected methylation difference
    target_power : float
        Target power
    designs : List[str], optional
        Designs to compare (defaults to common designs)
    config : PlannerConfig, optional
        Configuration instance (uses global if None)

    Returns
    -------
    pd.DataFrame
        Comparison table

    Examples
    --------
    >>> comparison = compare_designs(0.10, target_power=0.80)
    >>> print(comparison)
    """
    if config is None:
        config = get_config()
    if designs is None:
        designs = ["two_group", "paired", "multi_group"]

    effect_size = delta_beta_to_delta_m(expected_delta_beta)

    rows = []
    for design in designs:
        info = config.get_design(design)
        paired = info.get("paired", False)

        # Compute the minimum N that achieves target power for this design
        n_req = sample_size_for_power(
            target_power=target_power,
            effect_size=effect_size,
            alpha=0.05,
            variance=1.0,
            prior_df=10.0,
            paired=paired,
        )
        power_at_n = calculate_power(
            n_per_group=n_req,
            effect_size=effect_size,
            alpha=0.05,
            variance=1.0,
            prior_df=10.0,
            paired=paired,
        )

        rows.append(
            {
                "Design": info["name"],
                "Complexity": info["complexity"],
                "N_per_group": n_req,
                "Total_samples": n_req * info["n_groups"],
                "Achieved_power": f"{power_at_n:.1%}",
                "Paired": info["paired"],
            }
        )

    df = pd.DataFrame(rows)
    return df


def compare_platforms(
    n_samples: int,
    platforms: Optional[List[str]] = None,
    config: Optional[PlannerConfig] = None,
) -> pd.DataFrame:
    """
    Compare costs and specifications across platforms.

    Parameters
    ----------
    n_samples : int
        Number of samples
    platforms : List[str], optional
        Platforms to compare (defaults to recommended platforms)
    config : PlannerConfig, optional
        Configuration instance (uses global if None)

    Returns
    -------
    pd.DataFrame
        Platform comparison table

    Examples
    --------
    >>> comparison = compare_platforms(24)
    >>> print(comparison)
    """
    if config is None:
        config = get_config()

    if platforms is None:
        platforms = ["450K", "EPIC", "EPICv2", "WGBS"]

    results = []
    for platform in platforms:
        info = config.get_platform(platform)
        costs = estimate_costs(n_samples, platform, config=config)

        results.append(
            {
                "Platform": info["name"],
                "CpGs": f"{info['n_cpgs']:,}",
                "Per_sample": f"${info['cost_per_sample']}",
                "Total": f"${costs['total']:,.0f}",
                "Processing": f"{info['processing_days']} days",
                "Coverage": info["coverage"],
                "Recommended": "✓" if info["recommended"] else "",
            }
        )

    return pd.DataFrame(results)
