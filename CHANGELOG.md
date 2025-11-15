# CHANGELOG

All notable changes to this project will be documented in this file

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)

## [0.2.0] - 2025-11-15

### Added
- Study planning module (`core.planner`)
- Configuration database (`core.config`)
- Sample size and power analysis functions
- Cost estimation by platform
- Timeline projection with phase breakdown
- Batch planning and risk assessment
- Platform and design comparison utilities
- Comprehensive planning demo (`demos/planner_demo.py`)
- Modular documentation structure (`PLANNING.md`, `ANALYSIS.md`, `TROUBLESHOOTING.md`)

### Fixed
- Group mean calculation in chunked analysis
- Delta-beta to delta-M conversion approximation
- Paired design parameter propagation across all functions

## [0.1.0] - 2025-11-11

### Added
- Core differential methylation analysis engine (`core.engine`)
- Empirical Bayes variance shrinkage (Smyth's method)
- Memory-efficient chunked processing for large arrays
- Robust missing data handling with per-CpG fitting
- Batch effect adjustment via design matrix
- Comprehensive diagnostic visualizations
- PDF report generation with `PDFLogger`
- Complete analysis demo (`demos/engine_demo.py`)

### Features
- Support for 450K, EPIC, EPICv2, WGBS platforms
- Multiple shrinkage methods (auto, Smyth, median, fixed, none)
- Filtering by missingness and per-group counts
- Fast imputation (mean, median, KNN)
- Volcano plots, Q-Q plots, variance shrinkage plots
- Sample QC visualization (PCA, missingness, variance)
- Excel/CSV export

## Migration Guides

### v0.1.0 to v0.2.0 (Planning Module Addition)

**No breaking changes**. New functionality added:

```python
# Old workflow (still supported)
from methylation_engine.core.engine import fit_differential
results = fit_differential(M, design, ...)

# New workflow (planning + analysis)
from methylation_engine.core.planner import create_study_plan
from methylation_engine.core.engine import fit_differential

# Before data collection
plan = create_study_plan(
    study_name="My Study",
    expected_delta_beta=0.10,
    target_power=0.80
)

# After data collection
results = fit_differential(M, design, ...)
```

## Deprecation Policy

- **Major versions** (x.0.0) - May contain breaking changes
- **Minor versions** (1.x.0) - New features, no breaking changes
- **Patch versions** (1.0.x) - Bug fixes only

Deprecated features will be:

1. **Documented** in changelog
2. **Warned** via Python warnings
3. **Removed** after 2 minor versions minimum

Example:

```
v1.2.0 - Feature X deprecated (warning added)
v1.3.0 - Feature X still available (warning persists)
v1.4.0 - Feature X removed
```