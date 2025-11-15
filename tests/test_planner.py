#!/usr/bin/env python
# coding: utf-8

"""
Test suite for the study `planner` module.

The tests are designed to achieve ≥ 80% coverage of `core.planner`.

Usage:
    pytest -v --cov=core.planner --cov-report=html
"""

import shutil
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from methylation_engine.core.config import get_config
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
    sample_size_for_power,
)

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def temp_output_dir():
    """Create temporary directory for test outputs."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir)


@pytest.fixture
def test_config():
    """Get test configuration instance."""
    return get_config()


# ============================================================================
# TEST: EFFECT SIZE CONVERSION
# ============================================================================


class TestEffectSizeConversion:
    """Test delta_beta_to_delta_m function."""

    def test_basic_conversion(self):
        """Test basic Δβ to ΔM conversion."""
        delta_m = delta_beta_to_delta_m(0.10, baseline_beta=0.5)
        assert 0.5 < delta_m < 0.7  # Expected ~0.58

    def test_extreme_values(self):
        """Test clipping at boundaries."""
        # Large delta_beta should clip to 0.99
        delta_m = delta_beta_to_delta_m(0.90, baseline_beta=0.5)
        assert np.isfinite(delta_m)

        # Negative delta_beta
        delta_m_neg = delta_beta_to_delta_m(-0.20, baseline_beta=0.5)
        assert delta_m_neg < 0

    def test_zero_delta(self):
        """Zero change gives zero ΔM."""
        delta_m = delta_beta_to_delta_m(0.0, baseline_beta=0.5)
        assert abs(delta_m) < 1e-10


# ============================================================================
# TEST: POWER CALCULATION
# ============================================================================


class TestPowerCalculation:
    """Test calculate_power function."""

    def test_basic_power(self):
        """Power increases with sample size."""
        power_n10 = calculate_power(10, effect_size=1.5)
        power_n20 = calculate_power(20, effect_size=1.5)
        power_n50 = calculate_power(50, effect_size=1.5)

        assert power_n10 < power_n20 < power_n50
        assert 0 <= power_n10 <= 1

    def test_effect_size_impact(self):
        """Power increases with effect size."""
        power_small = calculate_power(12, effect_size=0.5)
        power_medium = calculate_power(12, effect_size=1.5)
        power_large = calculate_power(12, effect_size=3.0)

        assert power_small < power_medium < power_large

    def test_paired_vs_unpaired(self):
        """Paired designs have higher power."""
        power_unpaired = calculate_power(12, 1.5, paired=False)
        power_paired = calculate_power(12, 1.5, paired=True)

        assert power_paired > power_unpaired

    def test_alpha_level(self):
        """Lower alpha reduces power."""
        power_05 = calculate_power(12, 1.5, alpha=0.05)
        power_01 = calculate_power(12, 1.5, alpha=0.01)

        assert power_01 < power_05

    def test_two_sided_vs_one_sided(self):
        """Two-sided test has lower power."""
        power_two = calculate_power(12, 1.5, two_sided=True)
        power_one = calculate_power(12, 1.5, two_sided=False)

        assert power_two < power_one

    def test_variance_impact(self):
        """Higher variance reduces power."""
        power_low_var = calculate_power(12, 1.5, variance=0.5)
        power_high_var = calculate_power(12, 1.5, variance=2.0)

        assert power_high_var < power_low_var

    def test_prior_df_impact(self):
        """Prior df affects power through moderation."""
        power_low_df = calculate_power(12, 1.5, prior_df=5.0)
        power_high_df = calculate_power(12, 1.5, prior_df=20.0)

        # With shrinkage, higher prior df gives more conservative estimates
        assert abs(power_low_df - power_high_df) < 0.1  # Similar but not identical


# ============================================================================
# TEST: SAMPLE SIZE CALCULATION
# ============================================================================


class TestSampleSizeCalculation:
    """Test sample_size_for_power function."""

    def test_basic_sample_size(self):
        """Calculate sample size for target power."""
        n = sample_size_for_power(0.80, effect_size=1.5)

        assert n >= 3
        assert n < 100

        # Verify it achieves target power
        power = calculate_power(n, 1.5)
        assert power >= 0.80

    def test_higher_power_needs_more_samples(self):
        """Higher power requires larger N."""
        n_70 = sample_size_for_power(0.70, effect_size=1.5)
        n_80 = sample_size_for_power(0.80, effect_size=1.5)
        n_90 = sample_size_for_power(0.90, effect_size=1.5)

        assert n_70 < n_80 < n_90

    def test_larger_effect_needs_fewer_samples(self):
        """Larger effects need smaller samples."""
        n_small = sample_size_for_power(0.80, effect_size=0.5)
        n_large = sample_size_for_power(0.80, effect_size=2.0)

        assert n_large < n_small

    def test_paired_design(self):
        """Paired design needs fewer samples."""
        n_unpaired = sample_size_for_power(0.80, 1.5, paired=False)
        n_paired = sample_size_for_power(0.80, 1.5, paired=True)

        assert n_paired < n_unpaired

    def test_max_n_limit(self):
        """Returns max_n if target not achievable."""
        n = sample_size_for_power(0.99, effect_size=0.1, max_n=50)
        assert n == 50

    def test_minimum_n(self):
        """Always returns at least 3."""
        n = sample_size_for_power(0.01, effect_size=10.0)
        assert n >= 3


# ============================================================================
# TEST: DETECTABLE EFFECT SIZE
# ============================================================================


class TestDetectableEffectSize:
    """Test detectable_effect_size function."""

    def test_basic_detectable_effect(self):
        """Calculate minimum detectable effect."""
        effect = detectable_effect_size(12, target_power=0.80)

        assert effect > 0
        assert effect < 5.0

        # Verify it achieves target power
        power = calculate_power(12, effect)
        assert power >= 0.80

    def test_larger_n_detects_smaller_effects(self):
        """Larger samples detect smaller effects."""
        effect_n10 = detectable_effect_size(10, target_power=0.80)
        effect_n30 = detectable_effect_size(30, target_power=0.80)

        assert effect_n30 < effect_n10

    def test_paired_design(self):
        """Paired design detects smaller effects."""
        effect_unpaired = detectable_effect_size(12, paired=False)
        effect_paired = detectable_effect_size(12, paired=True)

        assert effect_paired < effect_unpaired

    def test_power_level(self):
        """Higher power requires larger detectable effect."""
        effect_70 = detectable_effect_size(12, target_power=0.70)
        effect_90 = detectable_effect_size(12, target_power=0.90)

        assert effect_70 < effect_90


# ============================================================================
# TEST: SAMPLE SIZE PLANNING
# ============================================================================


class TestSampleSizePlanning:
    """Test plan_sample_size function."""

    def test_basic_planning(self, test_config):
        """Basic sample size planning."""
        result = plan_sample_size(
            expected_delta_beta=0.10, target_power=0.80, config=test_config
        )

        assert "minimum" in result
        assert "recommended" in result
        assert "optimal" in result

        # Check increasing N across levels
        assert result["minimum"]["n_per_group"] <= result["recommended"]["n_per_group"]
        assert result["recommended"]["n_per_group"] <= result["optimal"]["n_per_group"]

    def test_required_parameters(self):
        """Must provide effect size."""
        with pytest.raises(ValueError, match="Must provide either"):
            plan_sample_size()

    def test_effect_size_conversion(self):
        """Convert delta_beta to effect_size."""
        result1 = plan_sample_size(expected_delta_beta=0.10)
        result2 = plan_sample_size(expected_effect_size=0.58)

        # Should give similar results
        assert (
            abs(
                result1["recommended"]["n_per_group"]
                - result2["recommended"]["n_per_group"]
            )
            <= 2
        )

    def test_design_types(self, test_config):
        """Different designs have different requirements."""
        result_two = plan_sample_size(
            expected_delta_beta=0.10, design_type="two_group", config=test_config
        )

        result_paired = plan_sample_size(
            expected_delta_beta=0.10, design_type="paired", config=test_config
        )

        # Paired needs fewer samples
        assert (
            result_paired["recommended"]["n_per_group"]
            < result_two["recommended"]["n_per_group"]
        )

    def test_power_levels(self):
        """Higher power requires more samples."""
        result_70 = plan_sample_size(expected_delta_beta=0.10, target_power=0.70)

        result_90 = plan_sample_size(expected_delta_beta=0.10, target_power=0.90)

        assert (
            result_90["recommended"]["n_per_group"]
            > result_70["recommended"]["n_per_group"]
        )

    def test_output_structure(self):
        """Check output contains expected fields."""
        result = plan_sample_size(expected_delta_beta=0.10)

        for level in ["minimum", "recommended", "optimal"]:
            assert "n_per_group" in result[level]
            assert "total_samples" in result[level]
            assert "power" in result[level]
            assert "detectable_effect_m" in result[level]
            assert "detectable_effect_beta" in result[level]

            # Power should be positive
            assert 0 < result[level]["power"] <= 1

    def test_minimum_n_enforced(self, test_config):
        """Respects minimum N from design."""
        result = plan_sample_size(
            expected_delta_beta=0.30,  # Large effect
            design_type="two_group",
            config=test_config,
        )

        design_info = test_config.get_design("two_group")
        assert result["recommended"]["n_per_group"] >= design_info["min_n_recommended"]

    def test_total_samples_calculation(self, test_config):
        """Total samples = n_per_group × n_groups."""
        result = plan_sample_size(
            expected_delta_beta=0.10, design_type="multi_group", config=test_config
        )

        design_info = test_config.get_design("multi_group")
        expected_total = result["recommended"]["n_per_group"] * design_info["n_groups"]
        assert result["recommended"]["total_samples"] == expected_total


# ============================================================================
# TEST: COST ESTIMATION
# ============================================================================


class TestCostEstimation:
    """Test estimate_costs function."""

    def test_basic_cost_estimation(self, test_config):
        """Basic cost estimation."""
        costs = estimate_costs(24, platform="EPIC", config=test_config)

        assert "breakdown" in costs
        assert "total" in costs
        assert "per_sample" in costs
        assert "n_samples" in costs
        assert "platform" in costs

        assert costs["total"] > 0
        assert costs["per_sample"] == costs["total"] / 24

    def test_cost_components(self, test_config):
        """Check cost components."""
        costs = estimate_costs(24, include_optional=True, config=test_config)

        breakdown = costs["breakdown"]
        assert "platform" in breakdown
        assert "dna_extraction" in breakdown
        assert "quality_control" in breakdown

        # Total should equal sum of components
        assert abs(costs["total"] - sum(breakdown.values())) < 0.01

    def test_optional_costs(self, test_config):
        """Optional costs can be excluded."""
        costs_with = estimate_costs(24, include_optional=True, config=test_config)
        costs_without = estimate_costs(24, include_optional=False, config=test_config)

        assert costs_with["total"] >= costs_without["total"]

    def test_sample_scaling(self, test_config):
        """Cost scales with sample size."""
        cost_10 = estimate_costs(10, platform="EPIC", config=test_config)
        cost_20 = estimate_costs(20, platform="EPIC", config=test_config)

        # Should be roughly 2x (per_sample costs scale linearly)
        ratio = cost_20["total"] / cost_10["total"]
        assert 1.8 < ratio < 2.2


# ============================================================================
# TEST: TIMELINE ESTIMATION
# ============================================================================


class TestTimelineEstimation:
    """Test estimate_timeline function."""

    def test_basic_timeline(self, test_config):
        """Basic timeline estimation."""
        timeline = estimate_timeline(24, platform="EPIC", config=test_config)

        assert "start_date" in timeline
        assert "completion_date" in timeline
        assert "total_duration_days" in timeline
        assert "total_duration_weeks" in timeline
        assert "total_duration_months" in timeline
        assert "phases" in timeline

        assert len(timeline["phases"]) > 0
        assert timeline["total_duration_days"] > 0

    def test_phase_structure(self, test_config):
        """Phases have required fields."""
        timeline = estimate_timeline(24, config=test_config)

        for phase in timeline["phases"]:
            assert "phase" in phase
            assert "start_date" in phase
            assert "end_date" in phase
            assert "duration_days" in phase
            assert "duration_weeks" in phase
            assert "description" in phase
            assert "critical" in phase

    def test_sample_size_scaling(self, test_config):
        """Timeline scales with sample size."""
        timeline_small = estimate_timeline(10, config=test_config)
        timeline_large = estimate_timeline(100, config=test_config)

        assert (
            timeline_large["total_duration_days"]
            > timeline_small["total_duration_days"]
        )

    def test_optional_phases(self, test_config):
        """Can exclude optional phases."""
        timeline_all = estimate_timeline(
            24, include_optional_phases=True, config=test_config
        )
        timeline_required = estimate_timeline(
            24, include_optional_phases=False, config=test_config
        )

        assert len(timeline_all["phases"]) > len(timeline_required["phases"])
        assert (
            timeline_all["total_duration_days"]
            > timeline_required["total_duration_days"]
        )

    def test_custom_start_date(self, test_config):
        """Can specify custom start date."""
        custom_date = datetime(2025, 6, 1)
        timeline = estimate_timeline(24, start_date=custom_date, config=test_config)

        assert timeline["start_date"] == custom_date

    def test_platform_differences(self, test_config):
        """Different platforms have different processing times."""
        timeline_epic = estimate_timeline(24, platform="EPIC", config=test_config)
        timeline_wgbs = estimate_timeline(24, platform="WGBS", config=test_config)

        # WGBS takes longer
        assert (
            timeline_wgbs["total_duration_days"] > timeline_epic["total_duration_days"]
        )

    def test_date_continuity(self, test_config):
        """Phase dates are continuous."""
        timeline = estimate_timeline(24, config=test_config)

        for i in range(len(timeline["phases"]) - 1):
            current_end = timeline["phases"][i]["end_date"]
            next_start = timeline["phases"][i + 1]["start_date"]
            assert current_end == next_start


# ============================================================================
# TEST: BATCH PLANNING
# ============================================================================


class TestBatchPlanning:
    """Test plan_batches function."""

    def test_single_batch(self):
        """Small studies fit in single batch."""
        plan = plan_batches(n_samples=24, n_groups=2)

        assert plan["n_batches"] == 1
        assert plan["balanced"] is True
        assert len(plan["distribution"]) == 1

    def test_multiple_batches(self):
        """Large studies need multiple batches."""
        plan = plan_batches(n_samples=200, n_groups=2)

        assert plan["n_batches"] > 1
        assert len(plan["distribution"]) == plan["n_batches"]

    def test_group_balancing(self):
        """Groups are balanced within batches."""
        plan = plan_batches(n_samples=96, n_groups=2)

        for batch in plan["distribution"]:
            assert "group1" in batch
            assert "group2" in batch
            assert batch["balance_ratio"] > 0

    def test_recommendations(self):
        """Provides recommendations for multiple batches."""
        plan = plan_batches(n_samples=200, n_groups=2)

        assert len(plan["recommendations"]) > 0
        assert any("batch" in rec.lower() for rec in plan["recommendations"])

    def test_batch_size_parameter(self):
        """Custom batch size."""
        plan = plan_batches(n_samples=100, samples_per_batch=50)

        assert plan["n_batches"] == 2

    def test_unbalanced_detection(self):
        """Detects unbalanced batches."""
        # Create scenario likely to be unbalanced
        plan = plan_batches(n_samples=97, n_groups=3)

        if not plan["balanced"]:
            assert any("unbalanced" in rec.lower() for rec in plan["recommendations"])

    def test_multi_group(self):
        """Handles multiple groups."""
        plan = plan_batches(n_samples=90, n_groups=3)

        for batch in plan["distribution"]:
            assert "group1" in batch
            assert "group2" in batch
            assert "group3" in batch


# ============================================================================
# TEST: RISK ASSESSMENT
# ============================================================================


class TestRiskAssessment:
    """Test assess_study_risks function."""

    def test_basic_risk_assessment(self, test_config):
        """Basic risk assessment."""
        risks = assess_study_risks(n_samples=24, power=0.80, config=test_config)

        assert "risks" in risks
        assert "n_risks" in risks
        assert "severity_counts" in risks
        assert "high_priority" in risks

        assert risks["n_risks"] == len(risks["risks"])

    def test_power_risks(self, test_config):
        """Low power triggers risk."""
        risks_low = assess_study_risks(n_samples=24, power=0.65, config=test_config)

        risks_high = assess_study_risks(n_samples=24, power=0.85, config=test_config)

        # Low power should have more risks
        assert risks_low["n_risks"] >= risks_high["n_risks"]

    def test_budget_risks(self, test_config):
        """Budget overrun triggers risk."""
        risks = assess_study_risks(
            n_samples=24,
            power=0.80,
            budget=10000,
            estimated_cost=15000,
            config=test_config,
        )

        # Should have budget risk
        budget_risks = [r for r in risks["risks"] if r["category"] == "Budget"]
        assert len(budget_risks) > 0

    def test_budget_severity_levels(self, test_config):
        """Budget risk severity scales with overrun."""
        # Minor overrun
        risks_minor = assess_study_risks(
            n_samples=24,
            power=0.80,
            budget=10000,
            estimated_cost=10500,
            config=test_config,
        )

        # Major overrun
        risks_major = assess_study_risks(
            n_samples=24,
            power=0.80,
            budget=10000,
            estimated_cost=20000,
            config=test_config,
        )

        minor_budget = [r for r in risks_minor["risks"] if r["category"] == "Budget"][0]
        major_budget = [r for r in risks_major["risks"] if r["category"] == "Budget"][0]

        # Both should exist but with different severities
        assert minor_budget["severity"] != major_budget["severity"]

    def test_sample_size_risks(self, test_config):
        """Small sample size triggers risk."""
        risks = assess_study_risks(n_samples=15, power=0.80, config=test_config)

        size_risks = [r for r in risks["risks"] if r["category"] == "Sample Size"]
        assert len(size_risks) > 0

    def test_batch_risks(self, test_config):
        """Multiple batches trigger risk."""
        risks = assess_study_risks(
            n_samples=200, power=0.80, n_batches=5, config=test_config
        )

        batch_risks = [r for r in risks["risks"] if "batch" in r["risk"].lower()]
        assert len(batch_risks) > 0

    def test_platform_risks(self, test_config):
        """Non-recommended platforms trigger risk."""
        risks = assess_study_risks(
            n_samples=24, power=0.80, platform="27K", config=test_config  # Discontinued
        )

        platform_risks = [r for r in risks["risks"] if r["category"] == "Platform"]
        assert len(platform_risks) > 0

    def test_severity_structure(self, test_config):
        """Risk structure contains required fields."""
        risks = assess_study_risks(n_samples=15, power=0.70, config=test_config)

        if risks["risks"]:
            risk = risks["risks"][0]
            assert "category" in risk
            assert "risk" in risk
            assert "severity" in risk
            assert "impact" in risk
            assert "mitigation" in risk
            assert risk["severity"] in ["Low", "Medium", "High"]


# ============================================================================
# TEST: COMPREHENSIVE STUDY PLAN
# ============================================================================


class TestComprehensiveStudyPlan:
    """Test create_study_plan function."""

    def test_basic_study_plan(self, test_config, temp_output_dir):
        """Create basic study plan."""
        plan = create_study_plan(
            study_name="Test Study",
            expected_delta_beta=0.10,
            config=test_config,
            output_dir=temp_output_dir,
        )

        assert "study_name" in plan
        assert "configuration" in plan
        assert "sample_size" in plan
        assert "costs" in plan
        assert "timeline" in plan
        assert "batch_plan" in plan
        assert "risk_assessment" in plan
        assert "created_date" in plan

    def test_no_output_dir(self, test_config):
        """Plan created without output directory."""
        plan = create_study_plan(
            study_name="Test Study",
            expected_delta_beta=0.10,
            config=test_config,
            output_dir=None,
        )

        # Should still have complete plan
        assert "sample_size" in plan
        assert "costs" in plan

    def test_budget_constraint(self, test_config):
        """Budget constraint affects platform selection."""
        plan = create_study_plan(
            study_name="Test Study",
            expected_delta_beta=0.10,
            platform="WGBS",  # Expensive
            budget=20000,  # Low budget
            config=test_config,
        )

        # May switch to cheaper platform
        if plan["configuration"]["platform_switched"]:
            assert plan["configuration"]["platform"] != "WGBS"
            assert plan["configuration"]["original_platform"] == "WGBS"

    def test_design_types(self, test_config):
        """Different designs produce valid plans."""
        for design in ["two_group", "paired", "multi_group"]:
            plan = create_study_plan(
                study_name=f"Test {design}",
                expected_delta_beta=0.10,
                design_type=design,
                config=test_config,
            )

            assert plan["configuration"]["design_type"] == design
            assert plan["sample_size"]["recommended"]["n_per_group"] > 0

    def test_configuration_stored(self, test_config):
        """Configuration parameters stored in plan."""
        plan = create_study_plan(
            study_name="Test Study",
            expected_delta_beta=0.10,
            target_power=0.85,
            alpha=0.01,
            platform="EPICv2",
            budget=30000,
            config=test_config,
        )

        cfg = plan["configuration"]
        assert cfg["expected_delta_beta"] == 0.10
        assert cfg["target_power"] == 0.85
        assert cfg["alpha"] == 0.01
        assert cfg["platform"] == "EPICv2"
        assert cfg["budget"] == 30000


# ============================================================================
# TEST: COMPARISON UTILITIES
# ============================================================================


class TestComparisonUtilities:
    """Test compare_designs and compare_platforms functions."""

    def test_compare_designs(self, test_config):
        """Compare multiple study designs."""
        comparison = compare_designs(
            expected_delta_beta=0.10,
            target_power=0.80,
            designs=["two_group", "paired", "multi_group"],
            config=test_config,
        )

        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 3
        assert "Design" in comparison.columns
        assert "N_per_group" in comparison.columns
        assert "Total_samples" in comparison.columns
        assert "Achieved_power" in comparison.columns

    def test_compare_designs_default(self, test_config):
        """Use default designs if none specified."""
        comparison = compare_designs(expected_delta_beta=0.10, config=test_config)

        assert len(comparison) >= 3  # At least default designs

    def test_compare_platforms(self, test_config):
        """Compare multiple platforms."""
        comparison = compare_platforms(
            n_samples=24, platforms=["EPIC", "EPICv2", "WGBS"], config=test_config
        )

        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 3
        assert "Platform" in comparison.columns
        assert "Platform" in comparison.columns
        assert "CpGs" in comparison.columns
        assert "Per_sample" in comparison.columns
        assert "Total" in comparison.columns
        assert "Processing" in comparison.columns
        assert "Coverage" in comparison.columns
        assert "Recommended" in comparison.columns

    def test_compare_platforms_default(self, test_config):
        """Use default designs if none specified."""
        comparison = compare_platforms(n_samples=24)

        assert len(comparison) >= 3  # At least default platforms
