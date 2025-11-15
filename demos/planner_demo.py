#!/usr/bin/env python
# coding: utf-8

"""
Study Planning Demo
Demonstrates comprehensive study design capabilities
"""

import os
import time
from datetime import datetime
import numpy as np
import pandas as pd

from methylation_engine.core.engine import PDFLogger

# Import configuration manager
from methylation_engine.core.config import get_config

from methylation_engine.core.planner import (
    # Core planning functions
    plan_sample_size,
    estimate_costs,
    estimate_timeline,
    plan_batches,
    assess_study_risks,
    create_study_plan,
    # Utility functions
    delta_beta_to_delta_m,
    calculate_power,
    sample_size_for_power,
    detectable_effect_size,
    quick_recommendation,
    # Comparison functions
    compare_designs,
    compare_platforms,
    # Visualization
    plot_power_curves,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "study_name": "Differential Methylation Discovery Study",
    "expected_delta_beta": 0.10,  # 10% methylation difference
    "target_power": 0.80,
    "alpha": 0.05,
    "platform": "EPIC",
    "budget": 25000,
    "variance": 1.0,
    "design_type": "paired"
}

# Configure output
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
report_path = f"log/{timestamp}"
os.makedirs(report_path, exist_ok=True)
pdf = PDFLogger(f"{report_path}/report.pdf", echo=True)

print("=" * 80)
pdf.log_text("# Differential Methylation Study Planner - Demo")
print("=" * 80)
pdf.log_text(f"**Study**: {CONFIG['study_name']}")
pdf.log_text(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
pdf.log_text(f"**Output directory**: `{report_path}`")
print()


design_info = get_config().get_design(CONFIG["design_type"])
platform_info = get_config().get_platform(CONFIG["platform"])

print("=" * 80)
pdf.log_text("## Study Configuration")
print("=" * 80)
pdf.log_text(f"- **Design type**: {design_info['name']}")
pdf.log_text(f"- **Paired design**: {'✔ Yes' if design_info['paired'] else '✗ No'}")
pdf.log_text(f"- **Platform**: {platform_info['name']}")
pdf.log_text(f"- **Expected Δβ**: {CONFIG['expected_delta_beta']:.2f}")
pdf.log_text(f"- **Target power**: {CONFIG['target_power']:.0%}")
pdf.log_text(f"- **Significance level (α)**: {CONFIG['alpha']}")
pdf.log_text(f"- **Budget**: ${CONFIG['budget']:,}" if CONFIG['budget'] else "- **Budget**: Not specified")
print()


# ============================================================================
# DEMO 1: EFFECT SIZE CONVERSION
# ============================================================================

print("\n" + "=" * 80)
pdf.log_text("## 1. Effect Size Conversion")
print("=" * 80)

start_time = time.time()

delta_beta = CONFIG["expected_delta_beta"]
delta_m = delta_beta_to_delta_m(delta_beta, baseline_beta=0.5)

pdf.log_text("### Effect Size Conversion:")
pdf.log_text(f"- Δβ = {delta_beta:.3f} (10% methylation difference)")
pdf.log_text(f"- ΔM = {delta_m:.2f} (M-value units)")
pdf.log_text("### Interpretation:")
pdf.log_text(f"- A {delta_beta*100:.0f}% change in methylation corresponds to")
pdf.log_text(f"- a {delta_m:.2f} unit change on the M-value scale.")

# Show conversion at different baseline levels
pdf.log_text("### Effect Size at Different Baseline Methylation Levels:")
baselines = [0.2, 0.5, 0.8]
for baseline in baselines:
    effect = delta_beta_to_delta_m(delta_beta, baseline_beta=baseline)
    pdf.log_text(f"- Baseline β = {baseline:.1f}: ΔM = {effect:.2f}")

elapsed = time.time() - start_time
pdf.log_text(f"✔ Completed in {elapsed:.2f} seconds")


# ============================================================================
# DEMO 2: POWER ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
pdf.log_text("## 2. Power Analysis")
print("=" * 80)

start_time_2 = time.time()

# Calculate power for different sample sizes
pdf.log_text("### Power Analysis for Different Sample Sizes:")
pdf.log_text(f"- **Effect size**: ΔM = {delta_m:.2f}")
pdf.log_text(f"- **Significance level**: α = {CONFIG['alpha']}")
pdf.log_text(f"- **Design**: {design_info['name']} ({'Paired' if design_info['paired'] else 'Unpaired'})")
print()

sample_sizes = [6, 10, 12, 15, 20, 30]
df = pd.DataFrame({"N per group": sample_sizes})
df["Total N"] = df["N per group"] * 2

df["Power"] = df["N per group"].apply(
    lambda n: calculate_power(
        n,
        delta_m,
        CONFIG["alpha"],
        CONFIG["variance"],
        paired=design_info["paired"]
    )
)

df["Interpretation"] = pd.cut(
    df["Power"],
    bins=[0, 0.7, 0.8, 0.9, 1],
    labels=["Underpowered", "Marginal", "Good", "Excellent"],
    include_lowest=True
)

df["Power"] = df["Power"].map("{:.1%}".format)

pdf.log_code(df.to_string(index=False, max_rows=20, max_cols=4))

# Find required sample size
sample_size_plan = plan_sample_size(
    expected_delta_beta=CONFIG["expected_delta_beta"],
    target_power=CONFIG["target_power"],
    alpha=CONFIG["alpha"],
    design_type=CONFIG["design_type"],
)
n_required = sample_size_plan['recommended']['n_per_group']

pdf.log_text(f"### ✔ Need {n_required} per group ({n_required*2} total) for {CONFIG['target_power']:.0%} power")

elapsed = time.time() - start_time_2
pdf.log_text(f"✔ Completed in {elapsed:.2f} seconds")


# ============================================================================
# DEMO 3: DETECTABLE EFFECT SIZES
# ============================================================================

print("\n" + "=" * 80)
pdf.log_text("## 3. Detectable Effect Sizes")
print("=" * 80)

start_time_3 = time.time()

pdf.log_text("### Minimum Detectable Effect for Different Sample Sizes:")
pdf.log_text(f"**Target power**: {CONFIG['target_power']:.0%}")
pdf.log_text(f"**Design**: {design_info['name']}")
print()

df = pd.DataFrame({
    "N per group": [8, 12, 16, 20, 30]
})

df["Min ΔM"] = df["N per group"].apply(lambda n: detectable_effect_size(n, CONFIG["target_power"], CONFIG["alpha"], paired=design_info['paired']))
df["Min Δβ (approx)"] = df["Min ΔM"] * 0.08

df["Interpretation"] = df["Min Δβ (approx)"].apply(
    lambda b: (
        "Can detect small effects" if b < 0.10
        else "Moderate effects only" if b < 0.15
        else "Large effects only"
    )
)

pdf.log_code(df.to_string(index=False, max_rows=20, max_cols=4))

elapsed = time.time() - start_time_3
pdf.log_text(f"✔ Completed in {elapsed:.2f} seconds")


# ============================================================================
# DEMO 4: SAMPLE SIZE PLANNING
# ============================================================================

print("\n" + "=" * 80)
print("## 4. Sample Size Planning")
print("=" * 80)

start_time_4 = time.time()

sample_size_plan = plan_sample_size(
    expected_delta_beta=CONFIG["expected_delta_beta"],
    target_power=CONFIG["target_power"],
    alpha=CONFIG["alpha"],
    design_type=CONFIG["design_type"],
)

pdf.log_text("### Sample Size Recommendations:")
print()

for level in ["minimum", "recommended", "optimal"]:
    plan = sample_size_plan[level]
    pdf.log_text(f"### {level.capitalize()}:")
    pdf.log_text(f"- **N per group**: {plan['n_per_group']}")
    pdf.log_text(f"- **Total samples**: {plan['total_samples']}")
    pdf.log_text(f"- **Statistical power**: {plan['power']:.1%}")
    pdf.log_text(f"- **Detectable Δβ**: {plan['detectable_effect_beta']:.3f}")
    pdf.log_text(f"- **Detectable ΔM**: {plan['detectable_effect_m']:.2f}")
    print()

pdf.log_text(f"### ✔ Recommended: {sample_size_plan['recommended']['n_per_group']} per group")

elapsed = time.time() - start_time_4
pdf.log_text(f"✔ Completed in {elapsed:.2f} seconds")


# ============================================================================
# DEMO 5: COST ESTIMATION
# ============================================================================

print("\n" + "=" * 80)
pdf.log_text("## 5. Cost Estimation")
print("=" * 80)

start_time_5 = time.time()

n_samples = sample_size_plan["recommended"]["total_samples"]
cost_estimate = estimate_costs(n_samples, platform=CONFIG["platform"])

pdf.log_text(f"### Cost Estimate for {n_samples} Samples:")
pdf.log_text(f"**Platform**: {CONFIG['platform']}")
print()

pdf.log_text("### Cost Breakdown:")
for category, cost in cost_estimate["breakdown"].items():
    pct = cost / cost_estimate["total"] * 100
    pdf.log_text(f"- **{category:<25}**: ${cost:>8,.0f}  ({pct:>5.1f}%)")

pdf.log_text(f"- **{'Total Cost':<25}**: ${cost_estimate['total']:>8,.0f}")
pdf.log_text(f"- **{'Per Sample':<25}**: ${cost_estimate['per_sample']:>8,.0f}")

if CONFIG["budget"]:
    pdf.log_text(f"### Budget Analysis:")
    pdf.log_text(f"- **Available budget**: ${CONFIG['budget']:,.0f}")
    pdf.log_text(f"- **Estimated cost**: ${cost_estimate['total']:,.0f}")
    
    if cost_estimate["total"] <= CONFIG["budget"]:
        remaining = CONFIG["budget"] - cost_estimate["total"]
        pdf.log_text(f"- **✔ Within budget (${remaining:,.0f} remaining)**")
    else:
        deficit = cost_estimate["total"] - CONFIG["budget"]
        pdf.log_text(f"- **✗ Over budget by ${deficit:,.0f}**")

elapsed = time.time() - start_time_5
pdf.log_text(f"✔ Completed in {elapsed:.2f} seconds")


# ============================================================================
# DEMO 6: TIMELINE ESTIMATION
# ============================================================================

print("\n" + "=" * 80)
pdf.log_text("## 6. Timeline Estimation")
print("=" * 80)

start_time_6 = time.time()

timeline = estimate_timeline(n_samples, platform=CONFIG["platform"])

pdf.log_text(f"### Project Timeline for {n_samples} Samples:")
pdf.log_text(f"- **Start**: {timeline['start_date'].strftime('%Y-%m-%d')}")
pdf.log_text(f"- **Estimated completion**: {timeline['completion_date'].strftime('%Y-%m-%d')}")
pdf.log_text(f"- **Total duration**: {timeline['total_duration_months']:.1f} months")
print()

pdf.log_text("### Phase Breakdown:")

# Convert timeline phases to DataFrame
df = pd.DataFrame(timeline["phases"])
df = df[["phase", "duration_weeks", "description"]]

# Rename columns for clarity and consistent header order
df = df.rename(columns={
    "phase": "Phase",
    "duration_weeks": "Duration (weeks)",
    "description": "Description"
})

# Ensure numeric formatting
df["Duration (weeks)"] = df["Duration (weeks)"].map(lambda x: f"{x:.1f}")

# Log as formatted string table
pdf.log_code(df.to_string(index=False, max_rows=20, max_cols=4))

elapsed = time.time() - start_time_6
pdf.log_text(f"✔ Completed in {elapsed:.2f} seconds")


# ============================================================================
# DEMO 7: BATCH PLANNING
# ============================================================================

print("\n" + "=" * 80)
pdf.log_text("## 7. Batch Planning")
print("=" * 80)

start_time_7 = time.time()

n_per_group = sample_size_plan["recommended"]["n_per_group"]
batch_plan = plan_batches(n_samples, n_per_group, samples_per_batch=96)

pdf.log_text(f"### Batch Plan for {n_samples} Samples:")
pdf.log_text(f"- **Number of batches**: {batch_plan['n_batches']}")
pdf.log_text(f"- **Balanced design**: {'✔ Yes' if batch_plan['balanced'] else '✗ No'}")
print()

if batch_plan["n_batches"] > 1:
    pdf.log_text("### Batch Distribution:")
    pdf.log_text(f"### {'Batch':<10} {'Total':<10} {'Group 1':<12} {'Group 2':<12} {'Balance'}")
    print("-" * 60)
    
    for batch in batch_plan["distribution"]:
        pdf.log_text(
            f"{batch['batch']:<10} "
            f"{batch['total_samples']:<10} "
            f"{batch['group1']:<12} "
            f"{batch['group2']:<12} "
            f"{batch['balance_ratio']:.2f}"
        )
    
    pdf.log_text("### Recommendations:")
    for rec in batch_plan["recommendations"]:
        pdf.log_text(f"- {rec}")
else:
    pdf.log_text("### ✔ All samples fit in single batch - no batch effects expected")

elapsed = time.time() - start_time_7
pdf.log_text(f"✔ Completed in {elapsed:.2f} seconds")


# ============================================================================
# DEMO 8: RISK ASSESSMENT
# ============================================================================

print("\n" + "=" * 80)
pdf.log_text("## 8. Risk Assessment")
print("=" * 80)

start_time_8 = time.time()

power = sample_size_plan["recommended"]["power"]
risks = assess_study_risks(
    n_samples=n_samples,
    power=power,
    budget=CONFIG["budget"],
    estimated_cost=cost_estimate["total"],
    n_batches=batch_plan["n_batches"],
    platform=CONFIG["platform"],
)

pdf.log_text(f"### Risk Assessment:")
pdf.log_text(f"**Total risks identified**: {risks['n_risks']}")
print()

for severity, count in risks["severity_counts"].items():
    if count > 0:
        pdf.log_text(f"- **{severity}**: {count}")

if risks["risks"]:
    pdf.log_text("### Detailed Risk Analysis:")
    print()
    
    for i, risk in enumerate(risks["risks"], 1):
        pdf.log_text(f"### {i}. [{risk['severity']}] {risk['category']}: {risk['risk']}")
        pdf.log_text(f"- **Impact**: {risk['impact']}")
        pdf.log_text(f"- **Mitigation**: {risk['mitigation']}")
        print()

if risks["high_priority"]:
    pdf.log_text(f"### ! {len(risks['high_priority'])} HIGH PRIORITY risks require attention")
else:
    pdf.log_text("### ✔ No high-priority risks identified")

elapsed = time.time() - start_time_8
pdf.log_text(f"✔ Completed in {elapsed:.2f} seconds")


# ============================================================================
# DEMO 9: DESIGN COMPARISON
# ============================================================================

print("\n" + "=" * 80)
pdf.log_text("## 9. Study Design Comparison")
print("=" * 80)

start_time_9 = time.time()

designs = ["two_group", "paired", "multi_group"]
comparison = compare_designs(
    CONFIG["expected_delta_beta"], 
    CONFIG["target_power"], 
    designs
)

pdf.log_text("### Comparing Study Designs:")
pdf.log_text(f"- **Effect size**: Δβ = {CONFIG['expected_delta_beta']:.2f}")
pdf.log_text(f"- **Target power**: {CONFIG['target_power']:.0%}")
pdf.log_text(f"- **Current study design**: {design_info['name']}")
print()

pdf.log_code(comparison.to_string(index=False, max_rows=20, max_cols=6))

pdf.log_text("### Key Insights:")
pdf.log_text(f"- Study uses {design_info['name'].lower()}, requiring {sample_size_plan['recommended']['n_per_group']} samples per group")
pdf.log_text("- Paired designs reduce N due to lower within-subject variance")
pdf.log_text("- Multi-group designs typically need more total samples to hit the same power.")
pdf.log_text("- Use paired design if feasible (before/after, matched samples)")

elapsed = time.time() - start_time_9
pdf.log_text(f"✔ Completed in {elapsed:.2f} seconds")


# ============================================================================
# DEMO 10: PLATFORM COMPARISON
# ============================================================================

print("\n" + "=" * 80)
pdf.log_text("## 10. Platform Comparison")
print("=" * 80)

start_time_10 = time.time()

platforms = ["450K", "EPIC", "EPICv2", "WGBS"]
platform_comparison = compare_platforms(n_samples, platforms)

pdf.log_text(f"### Comparing Platforms for {n_samples} Samples:")
print()

pdf.log_code(platform_comparison.to_string(index=False, max_rows=20, max_cols=7))

pdf.log_text("### Key Insights:")
pdf.log_text("- EPIC/EPICv2 recommended for discovery studies")
pdf.log_text("- 450K is legacy but still cost-effective")
pdf.log_text("- WGBS provides highest resolution but 2-3x more expensive")
pdf.log_text("- Consider coverage needs vs budget constraints")

elapsed = time.time() - start_time_10
pdf.log_text(f"\n✔ Completed in {elapsed:.2f} seconds")


# ============================================================================
# DEMO 11: VISUALIZATION
# ============================================================================

print("\n" + "=" * 80)
pdf.log_text("## 11. Visualization")
print("=" * 80)

start_time_11 = time.time()

# Power curves
print("Generating power curves...")
effect_sizes_to_use = [float(delta_m) * 0.5, float(delta_m), float(delta_m) * 1.5, float(delta_m) * 2, float(delta_m) * 2.5]
plot_power_curves(
    effect_sizes=effect_sizes_to_use[:4],
    sample_sizes=np.arange(3, 31),
    alpha=CONFIG["alpha"],
    paired=design_info['paired'],
    design_label=design_info['name'],
    save_path=f"{report_path}/power_curves.png",
)

pdf.log_image(
    f"{report_path}/power_curves.png",
    "Figure 1",
    "Power Analysis for Differential Methylation",
)

elapsed = time.time() - start_time_11
pdf.log_text(f"✔ Completed in {elapsed:.2f} seconds")


# ============================================================================
# DEMO 12: COMPREHENSIVE STUDY PLAN
# ============================================================================

print("\n" + "=" * 80)
pdf.log_text("## 12. Comprehensive Study Plan")
print("=" * 80)

start_time_12 = time.time()

print("Generating complete study plan...")

study_plan = create_study_plan(
    study_name=CONFIG["study_name"],
    expected_delta_beta=CONFIG["expected_delta_beta"],
    target_power=CONFIG["target_power"],
    alpha=CONFIG["alpha"],
    design_type=CONFIG["design_type"],
    platform=CONFIG["platform"],
    budget=CONFIG["budget"],
    output_dir=report_path,
)

pdf.log_text(f"✔ Study plan generated successfully")
pdf.log_text(f"### Output files:")
pdf.log_text(f"- `{report_path}/study_plan_report.txt`")
pdf.log_text(f"- `{report_path}/study_plan.xlsx`")
pdf.log_text(f"- `{report_path}/power_curves.png`")

elapsed = time.time() - start_time_12
pdf.log_text(f"✔ Completed in {elapsed:.2f} seconds")


# ============================================================================
# DEMO 13: QUICK RECOMMENDATIONS
# ============================================================================

print("\n" + "=" * 80)
pdf.log_text("## 13. Quick Recommendations")
print("=" * 80)

start_time_13 = time.time()

pdf.log_text("### Quick Recommendations for Different Scenarios:")
pdf.log_text(f"**(Using {design_info['name']})**")
print()

scenarios = [
    {"delta_beta": 0.05, "budget": 30000, "label": "Small effect, generous budget"},
    {"delta_beta": 0.10, "budget": 20000, "label": "Moderate effect, typical budget"},
    {"delta_beta": 0.15, "budget": 15000, "label": "Large effect, tight budget"},
    {"delta_beta": 0.20, "budget": None, "label": "Very large effect, no budget constraint"},
]

for i, scenario in enumerate(scenarios, 1):
    pdf.log_text(f"### {i}. {scenario['label']}:")
    
    rec = quick_recommendation(
        scenario["delta_beta"],
        budget=scenario["budget"],
        platform=CONFIG["platform"],
        design_type=CONFIG["design_type"],
    )
    
    rec_list = [r.strip() for r in rec.split(". ") if r.strip()]

    for r in rec_list:
        pdf.log_text(f"- {r}")

print()

elapsed = time.time() - start_time_13
pdf.log_text(f"✔ Completed in {elapsed:.2f} seconds")


# ============================================================================
# DEMO 14: DESIGN & PLATFORM INFO
# ============================================================================

print("\n" + "=" * 80)
pdf.log_text("## 14. Design & Platform Specifications")
print("=" * 80)

start_time_14 = time.time()

# Show available designs
pdf.log_text("### Available Study Designs:")
for design_name in ["two_group", "paired", "multi_group", "time_series", "factorial"]:
    info = get_config().get_design(design_name)
    pdf.log_text(f"### {info['name']}:")
    pdf.log_text(f"- **Description**: {info['description']}")
    pdf.log_text(f"- **Complexity**: {info['complexity']}")
    pdf.log_text(f"- **Paired**: {info['paired']}")
    pdf.log_text(f"- **Min N recommended**: {info['min_n_recommended']}")

# Show platform details
pdf.log_text("### Available Platforms:")
for platform_name in ["450K", "EPIC", "EPICv2", "WGBS", "RRBS"]:
    info = get_config().get_platform(platform_name)
    pdf.log_text(f"### {info['name']}:")
    pdf.log_text(f"- **CpGs**: {info['n_cpgs']:,}")
    pdf.log_text(f"- **Cost per sample**: ${info['cost_per_sample']}")
    pdf.log_text(f"- **Processing time**: {info['processing_days']} days")
    pdf.log_text(f"- **DNA required**: {info['dna_required_ng']} ng")
    pdf.log_text(f"- **Coverage**: {info['coverage']}")
    pdf.log_text(f"- **Recommended**: {'✔' if info['recommended'] else '✗'}")

elapsed = time.time() - start_time_14
pdf.log_text(f"\n✔ Completed in {elapsed:.2f} seconds")


# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
pdf.log_text("## Final Summary")
print("=" * 80)

total_time = time.time() - start_time

pdf.log_text(f"- **Study**: {CONFIG['study_name']}")
pdf.log_text(f"- **Expected effect**: Δβ = {CONFIG['expected_delta_beta']:.2f} (ΔM = {delta_m:.2f})")
pdf.log_text(f"- ✔ Total runtime: {total_time:.2f} seconds")
print()

pdf.log_text("### Key Recommendations:")
pdf.log_text(f"- ✔ Sample size: {sample_size_plan['recommended']['n_per_group']} per group ({sample_size_plan['recommended']['total_samples']} total)")
pdf.log_text(f"- ✔ Statistical power: {sample_size_plan['recommended']['power']:.1%}")
pdf.log_text(f"- ✔ Estimated cost: ${cost_estimate['total']:,.0f}")
pdf.log_text(f"- ✔ Timeline: {timeline['total_duration_months']:.1f} months")
pdf.log_text(f"- ✔ Batches: {batch_plan['n_batches']}")
print()

if risks["high_priority"]:
    pdf.log_text(f"- **!  HIGH PRIORITY**: {len(risks['high_priority'])} risks require attention")
    pdf.log_text("- Review full report for mitigation strategies")
    print()

pdf.log_text("### Next Steps:")
pdf.log_text("- Review full study plan report")
pdf.log_text("- Address any high-priority risks")
pdf.log_text("- Adjust design based on budget/timeline constraints")
pdf.log_text("- Finalize protocol and begin IRB submission")
pdf.log_text("- Use `core.engine` for data analysis once data collected")
print()

# Save PDF
pdf.save()

pdf.log_text(f"**All outputs saved to**: {report_path}/")
print()

print("=" * 80)
pdf.log_text("## Demo Complete!")
print("=" * 80)
pdf.log_text(f"**Total runtime**: {total_time:.1f}s")
print("=" * 80)