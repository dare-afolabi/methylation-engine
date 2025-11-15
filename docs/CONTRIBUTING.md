# Contributing Guide

Thank you for considering contributing to the Differential Methylation Analysis Pipeline. This document provides guidelines for contributing

## Quick Start

1. **Fork and clone**
```bash
git clone https://github.com/dare-afolabi/methylation-engine.git
cd methylation-engine
```

2. **Create branch**

```bash
git checkout -b feature/my-feature
```

3. **Install dev dependencies**

```bash
pip install -e ".[dev]"
```

4. **Make changes and test**

```bash
pytest tests/
```

5. **Submit PR**

## Development Setup

### Requirements

- Python 3.9+
- Git
- pytest, isort, black, flake8, mypy (installed via `pip install -e ".[dev]"`)

### Project Structure

```
methylation-engine/
├── core/
│   ├── __init__.py
│   ├── config.py       # Platform/design database
│   ├── planner.py      # Study planning functions
│   └── engine.py       # Statistical analysis
├── demos/
│   ├── planner_demo.py # Planning workflow
│   └── engine_demo.py  # Analysis workflow
├── tests/
│   ├── test_planner.py
│   └── test_engine.py
└── docs/
    ├── PLANNING.md
    ├── ANALYSIS.md
    ├── CONTRIBUTING.md
    ├── CODE_OF_CONDUCT.md
    └── TROUBLESHOOTING.md
```

## What to Contribute

### High Priority

- **Bug fixes** - See [GitHub Issues](https://github.com/dare-afolabi/methylation-engine/issues)
- **Documentation improvements** - Clarifications, examples
- **Test coverage** - Expand unit tests
- **Performance optimizations** - Faster algorithms

### Features

- **New platforms** - Add to `config.py`
- **Additional study designs** - Extend planning module
- **Visualization enhancements** - New diagnostic plots
- **File format support** - Import/export formats
- **More**

### Not Accepting

- Features that increase complexity without clear benefit
- Code that duplicates existing functionality

## Code Standards

### Style Guide

- **PEP 8** for Python code
- **Black** for formatting (line length 88)
- **Type hints** for public functions
- **Docstrings** in NumPy format

### Example Function

```python
def calculate_power(
    n_per_group: int,
    effect_size: float,
    alpha: float = 0.05,
    paired: bool = False
) -> float:
    """
    Calculate statistical power.
    
    Parameters
    ----------
    n_per_group : int
        Sample size per group
    effect_size : float
        Expected effect (M-value units)
    alpha : float
        Significance level
    paired : bool
        Whether design is paired
    
    Returns
    -------
    float
        Power (0-1)
    
    Examples
    --------
    >>> calculate_power(12, 1.5, alpha=0.05)
    0.82
    """
    # Implementation
    pass
```

### Formatting

```bash
# Lint imports
isort core/ tests/

# Auto-format code
black core/ tests/

# Check lint style
flake8 core/ tests/

# Type check
mypy core/
```

## Testing

### Run Tests

```bash
# All tests
pytest

# Specific module
pytest tests/test_planner.py

# With coverage
pytest --cov=core --cov-report=html
```

### Writing Tests

```python
# tests/test_calculate_power.py
import pytest
import numpy as np
from methylation_engine.core.planner import calculate_power

def test_calculate_power_basic():
    """Test power calculation with known values."""
    power = calculate_power(
        n_per_group=12,
        effect_size=1.5,
        alpha=0.05
    )
    assert 0.7 < power < 0.9

def test_calculate_power_paired():
    """Paired designs should need fewer samples."""
    power_unpaired = calculate_power(12, 1.5, paired=False)
    power_paired = calculate_power(12, 1.5, paired=True)
    assert power_paired > power_unpaired

def test_calculate_power_invalid_input():
    """Test error handling."""
    with pytest.raises(ValueError):
        calculate_power(n_per_group=0, effect_size=1.5)
```

### Test Requirements

- **Unit tests** for all public functions
- **Integration tests** for workflows
- **Edge cases** (empty data, invalid inputs)
- **Performance tests** for large datasets (optional)

## Documentation

### Docstring Format

Use **NumPy style** docstrings:

```python
def function_name(param1, param2):
    """
    Short description (one line).
    
    Longer description if needed. Can span
    multiple paragraphs.
    
    Parameters
    ----------
    param1 : type
        Description of param1
    param2 : type, optional
        Description of param2 (default: value)
    
    Returns
    -------
    return_type
        Description of return value
    
    Raises
    ------
    ValueError
        When param1 is invalid
    
    Examples
    --------
    >>> function_name(1, 2)
    3
    
    Notes
    -----
    Additional implementation details.
    
    References
    ----------
    [1] Author et al. (2024). Paper title. Journal.
    """
```

### User Documentation

When adding features, update:

- **PLANNING.md** or **ANALYSIS.md** - Add usage examples
- **TROUBLESHOOTING.md** - Add common issues (if applicable)

Keep docs **concise** - prefer examples over lengthy explanations.

## Pull Request Process

### Before Submitting

- [ ] Code follows style guide (`isort`, `black`, `flake8`, `mypy`)
- [ ] Tests pass (`pytest`)
- [ ] New functions have docstrings
- [ ] User docs updated (if needed)
- [ ] Commit messages are clear

### PR Template

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
- [ ] Unit tests added/updated
- [ ] All tests pass
- [ ] Tested on example data

## Checklist
- [ ] Code formatted (black)
- [ ] Docstrings added
- [ ] Documentation updated
- [ ] No breaking changes (or discussed in issue)

## Related Issues
Closes #123
```

### Review Process

1. **Automated checks** - CI runs tests, style checks
2. **Maintainer review** - Code quality, design
3. **Revisions** - Address feedback
4. **Merge** - Squash commits to keep history clean

-----

## Commit Messages

### Format

```
type(scope): brief description

Longer explanation if needed.

Fixes #123
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code formatting (no logic change)
- `refactor`: Code restructuring
- `test`: Test updates
- `perf`: Performance improvement

### Examples

```
feat(planner): add factorial design support

Add factorial design configuration and power adjustment
factor to config.py. Update plan_sample_size() to handle
2-way factorial designs.

Closes #45

---

fix(engine): correct group mean calculation in chunked analysis

Previously, chunked analysis failed to add group means when
using subset of M. Now passes full M matrix to ensure all
result CpGs are present.

Fixes #67

---

docs(planning): add timeline estimation examples

Add code examples for estimate_timeline() to PLANNING.md.
Include examples of optional phase control and batch
adjustments.
```

## Adding New Platforms

To add a new methylation platform:

### 1. Add to `config.py`

```python
DEFAULT_PLATFORMS = {
    # ... existing platforms ...
    'NewArray': {
        'name': 'New Methylation Array',
        'manufacturer': 'Company',
        'n_cpgs': 1000000,
        'cost_per_sample': 550,
        'processing_days': 7,
        'dna_required_ng': 200,
        'coverage': 'Enhanced',
        'release_year': 2025,
        'status': 'Current',
        'recommended': True,
        'notes': 'Description of platform'
    }
}
```

### 2. Update Documentation

- Add to comparison tables in `PLANNING.md`
- Update platform list in `README.md`

### 3. Add Test

```python
def test_new_platform():
    from methylation_engine.core.config import get_config
    config = get_config()
    platform = config.get_platform('NewArray')
    assert platform['n_cpgs'] == 1000000
    assert platform['cost_per_sample'] == 550
```

## Adding New Study Designs

To add a new study design:

### 1. Add to `config.py`

```python
DEFAULT_DESIGNS = {
    # ... existing designs ...
    'new_design': {
        'name': 'New Design Name',
        'description': 'Description of design',
        'n_groups': 2,  # or more
        'paired': False,
        'complexity': 'Simple',  # or 'Moderate', 'Complex'
        'min_n_recommended': 12,
        'power_adjustment': 1.0,  # Adjust based on efficiency
        'analysis_method': 'Statistical method used',
        'example_uses': ['Use case 1', 'Use case 2']
    }
}
```

### 2. Update Planning Functions

Ensure `plan_sample_size()` handles the new design correctly.

### 3. Add Tests

```python
def test_new_design():
    from methylation_engine.core.planner import plan_sample_size
    result = plan_sample_size(
        expected_delta_beta=0.10,
        design_type='new_design'
    )
    assert result['recommended']['n_per_group'] > 0
```

### 4. Update Documentation

- Add to design comparison tables
- Add example to `PLANNING.md`

## Performance Guidelines

### Optimization Priorities

1. **Correctness first** - Never sacrifice accuracy for speed
2. **Profile before optimizing** - Measure, don’t guess
3. **Vectorize operations** - Use NumPy/pandas operations
4. **Minimize copies** - Avoid unnecessary data duplication

### Benchmarking

```python
import time
import numpy as np

def benchmark_function():
    """Time a function with large input."""
    M = np.random.randn(100000, 20)
    
    start = time.time()
    result = my_function(M)
    elapsed = time.time() - start
    
    print(f"Time: {elapsed:.2f}s")
    print(f"Rate: {len(M)/elapsed:.0f} CpGs/sec")
```

### Performance Tests

Add performance tests for large datasets:

```python
@pytest.mark.slow
def test_chunked_analysis_performance():
    """Ensure chunked method handles 450K arrays in <5 min."""
    M = generate_large_dataset(450000, 20)
    
    start = time.time()
    results = fit_differential_chunked(M, design, chunk_size=50000)
    elapsed = time.time() - start
    
    assert elapsed < 300  # 5 minutes
    assert len(results) > 0
```

## Release Process

Maintainers only:

1. **Update version** in `core/__init__.py`
2. **Update CHANGELOG.md**
3. **Tag release**

```bash
git tag -a v0.2.1 -m "Release v0.2.1"
git push origin v0.2.1
```

1. **GitHub Release** - Auto-generates from tag
2. **PyPI upload** (if/when published)

-----

## Questions?

- **General questions**: [GitHub Discussions](https://github.com/dare-afolabi/methylation-engine/discussions/4)
- **Bug reports**: [GitHub Issues](https://github.com/dare-afolabi/methylation-engine/issues)
- **Feature requests**: [GitHub Issues](https://github.com/dare-afolabi/methylation-engine/issues)
- **Security issues**: Email [dare.afolabi@outlook.com](mailto:dare.afolabi@outlook.com) privately

## License

By contributing, you agree your contributions will be licensed under the MIT License.