# Study Planning Guide

Complete guide to designing DNA methylation studies with optimal power and budget

## Quick Start

```python
from methylation_engine.core.planner import create_study_plan

plan = create_study_plan(
    study_name="My Study",
    expected_delta_beta=0.10,
    target_power=0.80,
    platform='EPIC',
    budget=25000,
    design_type='paired',
    output_dir='output'
)
```

Generates:

- Sample size recommendations
- Cost breakdown
- Timeline with phases
- Batch planning
- Risk assessment
- Power curves
- Excel and text reports

## Core Concepts

### Effect Sizes

**Delta-beta (Δβ)**: Methylation difference on 0-1 scale

- Small: 0.05 (5% difference)
- Medium: 0.10 (10% difference)
- Large: 0.20 (20% difference)

**Delta-M (ΔM)**: Log2 fold change in M-values

- Automatically converted from Δβ
- Used internally for power calculations

```python
from methylation_engine.core.planner import delta_beta_to_delta_m

effect = delta_beta_to_delta_m(0.10)  # 0.58 M-value units
```

### Study Designs

|Design     |Description        |Power Adjustment|Min N|
|-----------|-------------------|----------------|-----|
|Two-Group  |Independent samples|1.0x            |12   |
|Paired     |Matched samples    |0.71x           |10   |
|Multi-Group|3+ groups          |1.2x            |15   |
|Time Series|Multiple timepoints|0.85x           |8    |
|Factorial  |2+ factors         |1.5x            |20   |

### Platforms

|Platform|CpGs|Cost/Sample|Coverage   |
|--------|----|-----------|-----------|
|450K    |485K|$400       |Genome-wide|
|EPIC    |867K|$500       |Enhanced   |
|EPICv2  |935K|$550       |Improved   |
|WGBS    |28M |$1200      |Complete   |

-----

## Workflow

### 1. Sample Size Planning

```python
from methylation_engine.core.planner import plan_sample_size

result = plan_sample_size(
    expected_delta_beta=0.10,
    target_power=0.80,
    alpha=0.05,
    design_type='paired'
)

# Three options provided
print(result['minimum'])      # Conservative
print(result['recommended'])  # Target power
print(result['optimal'])      # Extra power
```

**Output**:

```python
{
    'n_per_group': 24,
    'total_samples': 48,
    'power': 0.804,
    'detectable_effect_beta': 0.048
}
```

### 2. Cost Estimation

```python
from methylation_engine.core.planner import estimate_costs

costs = estimate_costs(
    n_samples=48,
    platform='EPIC',
    include_optional=True
)

print(f"Total: ${costs['total']:,}")
print(f"Per sample: ${costs['per_sample']:,}")
```

**Breakdown**:

- Platform arrays
- DNA extraction
- Quality control
- Data storage
- Bioinformatics
- Project management
- Optional: Validation

### 3. Timeline Projection

```python
from methylation_engine.core.planner import estimate_timeline

timeline = estimate_timeline(
    n_samples=48,
    platform='EPIC',
    include_optional_phases=True
)

print(f"Duration: {timeline['total_duration_months']:.1f} months")

for phase in timeline['phases']:
    print(f"{phase['phase']}: {phase['duration_weeks']:.1f} weeks")
```

**Phases**:

1. Planning & IRB
2. Sample Collection
3. DNA Extraction
4. Array Processing
5. Data Generation
6. Quality Control
7. Analysis
8. Validation (optional)

### 4. Batch Planning

```python
from methylation_engine.core.planner import plan_batches

batch_plan = plan_batches(
    n_samples=48,
    n_groups=2,
    samples_per_batch=96
)

print(f"Batches: {batch_plan['n_batches']}")
print(f"Balanced: {batch_plan['balanced']}")
```

**Recommendations**:

- Balance groups across batches
- Include batch in design matrix
- Use same reagent lots
- Add technical replicates

### 5. Risk Assessment

```python
from methylation_engine.core.planner import assess_study_risks

risks = assess_study_risks(
    n_samples=48,
    power=0.80,
    budget=25000,
    estimated_cost=36000,
    n_batches=1,
    platform='EPIC'
)

for risk in risks['high_priority']:
    print(f"[{risk['severity']}] {risk['risk']}")
    print(f"  Mitigation: {risk['mitigation']}")
```

## Comparisons

### Compare Designs

```python
from methylation_engine.core.planner import compare_designs

comparison = compare_designs(
    expected_delta_beta=0.10,
    target_power=0.80,
    designs=['two_group', 'paired', 'multi_group']
)

print(comparison)
```

**Shows**:

- Required N per design
- Total samples needed
- Achieved power
- Design complexity

### Compare Platforms

```python
from methylation_engine.core.planner import compare_platforms

comparison = compare_platforms(
    n_samples=48,
    platforms=['EPIC', 'EPICv2', 'WGBS']
)

print(comparison)
```

**Shows**:

- Total cost per platform
- CpG coverage
- Processing time
- Recommendations

## Quick Recommendations

```python
from methylation_engine.core.planner import quick_recommendation

rec = quick_recommendation(
    expected_delta_beta=0.10,
    budget=20000,
    platform='EPIC',
    design_type='paired'
)

print(rec)
# "Recommend 24 per group (48 total) for 80% power (Paired Design). 
#  Cost: $36,000. Exceeds budget by $16,000."
```

## Configuration

### Custom Platforms

```python
from methylation_engine.core.config import get_config

config = get_config()

config.add_custom_platform('CustomArray', {
    'name': 'My Custom Array',
    'n_cpgs': 1000000,
    'cost_per_sample': 600,
    'processing_days': 7,
    'coverage': 'Custom',
    'recommended': True
})
```

### Regional Pricing

```python
from methylation_engine.core.config import update_regional_pricing

# Adjust all prices by 30% for European costs
update_regional_pricing('EU', multiplier=1.30)
```

### Save/Load Configurations

```python
from methylation_engine.core.config import get_config, load_config

# Export template
config = get_config()
config.export_to_excel('my_config.xlsx')

# Edit in Excel, then load
load_config('my_config.xlsx')
```

## Visualization

```python
from methylation_engine.core.planner import plot_power_curves

# Power curves for different effect sizes
plot_power_curves(
    effect_sizes=[0.5, 1.0, 1.5, 2.0],
    paired=True,
    design_label='Paired Design',
    save_path='power_curves.png'
)

```

## Best Practices

### Sample Size

- **Target 80% power** minimum
- **Add 10-15% oversample** for QC failures
- **Use paired designs** when possible (lower N required)

### Budget

- **Include 10-20% contingency** for unexpected costs
- **Compare platforms** before finalizing
- **Consider phased approach** if over budget

### Timeline

- **Add buffer time** to each phase (10-20%)
- **Account for holidays** and equipment downtime
- **Plan validation early** if required

### Batch Effects

- **Balance groups** across batches
- **Randomize assignments**
- **Include batch in analysis** design matrix
- **Use technical replicates** across batches

## Examples

See `demos/planner_demo.py` for:

- Effect size conversions
- Power analysis at different N
- Detectable effects
- Complete study plan generation
- Design and platform comparisons
- Visualization examples

## Troubleshooting

See [Troubleshooting](https://github.com/dare-afolabi/methylation-engine/blob/main/docs/TROUBLESHOOTING.md)

## References

- Jung, S.H., Young, S.S. (2012). Power and sample size calculation for microarray studies. *Journal of Biopharmaceutical Statistics*, 22(1):30-42.
- Liu, P., & Hwang, J.T.G. (2007). Quick calculation for sample size while controlling false discovery rate with application to microarray analysis. *Bioinformatics*, 23(6), 739–746.
- Du, P., Zhang, X., Huang, C.-C., Jafari, N., Kibbe, W.A., Hou, L., & Lin, S. (2010). Comparison of Beta-value and M-value methods for quantifying methylation levels by microarray analysis. *BMC Bioinformatics*, 11:587.