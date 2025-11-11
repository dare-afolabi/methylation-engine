# Contributing Guide

Thank you for considering contributing to the Differential Methylation Pipeline! This document provides guidelines for contributing.

-----

## Ways to Contribute

### 1. Report Bugs

- Use [GitHub Issues](https://github.com/dare-afolabi/methylation-engine/issues)
- Include minimal reproducible example
- Specify Python version and dependencies

### 2. Suggest Features

- Open a [Discussion](https://github.com/discussions/) first
- Explain use case and benefits
- Consider implementation complexity

### 3. Improve Documentation

- Fix typos or unclear sections
- Add examples or tutorials
- Translate documentation

### 4. Submit Code

- Bug fixes
- New features
- Performance improvements
- Test coverage

-----

## Development Setup

### 1. Fork and Clone

```bash
# Fork on GitHub, then:
git clone https://github.com/YOUR_USERNAME/methylation-engine.git
cd methylation-engine
git remote add upstream https://github.com/dare-afolabi/methylation-engine.git
```

### 2. Create Environment

```bash
# Using conda
conda create -n engine python=3.10
conda activate engine

# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Development Dependencies

```bash
pip install -e ".[dev]"  # includes pytest, black, etc.
```

### 4. Install Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
```

-----

## Code Standards

### Style Guide

**Follow PEP 8** with these specifics:

- Line length: 88 characters (Black default)
- Use type hints for function signatures
- Docstrings in NumPy style

**Example**:

```python
def fit_differential(
    M: pd.DataFrame,
    design: pd.DataFrame,
    contrast: Optional[np.ndarray] = None,
    shrink: Union[str, float] = "auto"
) -> pd.DataFrame:
    """
    Fit linear models with empirical Bayes shrinkage.
    
    Parameters
    ----------
    M : pd.DataFrame
        CpG x samples matrix of M-values
    design : pd.DataFrame
        Design matrix (samples x covariates)
    contrast : np.ndarray, optional
        Contrast vector for testing
    shrink : str or float
        Shrinkage method
        
    Returns
    -------
    pd.DataFrame
        Results with columns: logFC, t, pval, padj, etc.
        
    Examples
    --------
    >>> M = pd.DataFrame(np.random.randn(100, 10))
    >>> design = pd.DataFrame({'group': [0]*5 + [1]*5})
    >>> res = fit_differential(M, design, contrast=np.array([0, 1]))
    """
    # Implementation
```

### Formatting

Use **Black** for code formatting:

```bash
black core/ tests/
```

Use **isort** for import sorting:

```bash
isort core/ tests/
```

### Linting

```bash
flake8 core/ tests/
mypy core/
```

-----

## Testing

### Writing Tests

Place tests in `tests/` directory:

```python
# tests/test_engine.py
import numpy as np
import pandas as pd
import pytest
from core.engine import fit_differential

def test_fit_differential_basic():
    """Test basic two-group comparison"""
    # Setup
    M = pd.DataFrame(
        np.random.randn(100, 10),
        index=[f'cg{i:08d}' for i in range(100)],
        columns=[f'S{i}' for i in range(10)]
    )
    design = pd.DataFrame({
        'Intercept': 1,
        'Group': [0]*5 + [1]*5
    }, index=M.columns)
    
    # Execute
    results = fit_differential(
        M, design,
        contrast=np.array([0, 1]),
        shrink='none'
    )
    
    # Assert
    assert 'logFC' in results.columns
    assert 'pval' in results.columns
    assert len(results) == 100
    assert results['pval'].min() >= 0
    assert results['pval'].max() <= 1


def test_fit_differential_with_missing():
    """Test handling of missing data"""
    M = pd.DataFrame(np.random.randn(50, 8))
    M.iloc[0:10, 0:2] = np.nan  # Add missing
    
    design = pd.DataFrame({
        'Intercept': 1,
        'Group': [0]*4 + [1]*4
    }, index=M.columns)
    
    results = fit_differential(
        M, design,
        contrast=np.array([0, 1]),
        min_count=3
    )
    
    assert len(results) <= 50  # Some may be filtered
    assert results['n_obs'].min() >= 3


def test_fit_differential_invalid_design():
    """Test that invalid design raises error"""
    M = pd.DataFrame(np.random.randn(10, 5))
    design = pd.DataFrame({
        'Intercept': 1,
        'Group': [0, 1, 2]  # Wrong length!
    })
    
    with pytest.raises(ValueError, match="design rows"):
        fit_differential(M, design, contrast=np.array([0, 1]))
```

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=core --cov-report=html

# Specific test file
pytest tests/test_engine.py

# Specific test
pytest tests/test_engine.py::test_fit_differential_basic

# Verbose
pytest -v
```

### Test Coverage Goals

- **Core functions: >90%
- **Utilities: >80%
- **Visualization: >60%

-----

## Pull Request Process

### 1. Create Feature Branch

```bash
git checkout -b feature/my-new-feature
# or
git checkout -b fix/bug-description
```

### 2. Make Changes

- Write code
- Add tests
- Update documentation
- Run tests locally

### 3. Commit Changes

Use conventional commits:

```bash
git commit -m "feat: add support for paired samples"
git commit -m "fix: correct p-value calculation in F-test"
git commit -m "docs: update quickstart guide"
git commit -m "test: add tests for missing data handling"
```

**Commit types:**

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Formatting, no code change
- `refactor`: Code restructure, no behavior change
- `test`: Adding tests
- `perf`: Performance improvement

### 4. Push and Create PR

```bash
git push origin feature/my-new-feature
```

Then open Pull Request on GitHub.

### 5. PR Checklist

- [ ] Tests pass locally
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] Code formatted with Black
- [ ] No linting errors
- [ ] CHANGELOG.md updated (for features/fixes)
- [ ] Linked to relevant issue

### 6. Review Process

- Maintainers will review within 1 week
- Address feedback in new commits
- Once approved, maintainer will merge

-----

## Code Review Guidelines

### For Authors

- Keep PRs focused and small (<500 lines)
- Write clear PR description
- Respond to feedback promptly
- Don’t force-push after review starts

### For Reviewers

- Be constructive and specific
- Suggest improvements, don’t demand perfection
- Approve if code is correct and maintainable
- Nitpicks should be marked as such

**Review checklist:**

- [ ] Code is correct and handles edge cases
- [ ] Tests are comprehensive
- [ ] Documentation is clear
- [ ] Performance is acceptable
- [ ] Style follows guidelines

-----

## Adding New Features

### Typical Workflow

1. **Discuss first** - Open issue/discussion
1. **Design** - Consider API, edge cases, tests
1. **Implement** - Write code incrementally
1. **Test** - Comprehensive test coverage
1. **Document** - Docstrings, README, examples
1. **Submit PR** - Follow checklist above

### Example: Adding New Shrinkage Method

```python
# 1. Add to fit_differential
def fit_differential(
    M: pd.DataFrame,
    design: pd.DataFrame,
    shrink: Union[str, float] = "auto",
    ...
):
    # ... existing code ...
    
    # Add new option
    elif shrink == "empirical_bayes_v2":
        d0, s0sq = _estimate_eb_v2(s2, df_resid)
        s2_post = (df_resid * s2 + d0 * s0sq) / (df_resid + d0)
        df_total = df_resid + d0
    
    # ... rest of code ...


# 2. Implement helper function
def _estimate_eb_v2(
    s2: np.ndarray,
    df_resid: float
) -> Tuple[float, float]:
    """
    Estimate prior using new EB method.
    
    Parameters
    ----------
    s2 : np.ndarray
        Raw variances
    df_resid : float
        Residual df
        
    Returns
    -------
    Tuple[float, float]
        (d0, s0_squared)
        
    References
    ----------
    Author et al. (2025). Journal. DOI: xxx
    """
    # Implementation
    pass


# 3. Add tests
def test_fit_differential_eb_v2():
    """Test new EB shrinkage method"""
    M = pd.DataFrame(np.random.randn(100, 10))
    design = pd.DataFrame({'Group': [0]*5 + [1]*5})
    
    results = fit_differential(
        M, design,
        contrast=np.array([0, 1]),
        shrink='empirical_bayes_v2'
    )
    
    assert 'd0' in results.columns
    assert results['d0'].iloc[0] > 0


# 4. Update documentation
# - Add to README
# - Update docstring
# - Add to CHANGELOG
```

-----

## Documentation

### Docstring Format

Use **NumPy style**:

```python
def function(param1: int, param2: str = "default") -> bool:
    """
    Short one-line description.
    
    Longer description explaining what the function does,
    when to use it, and any important details.
    
    Parameters
    ----------
    param1 : int
        Description of param1
    param2 : str, optional
        Description of param2 (default: "default")
        
    Returns
    -------
    bool
        Description of return value
        
    Raises
    ------
    ValueError
        If param1 is negative
        
    Examples
    --------
    >>> result = function(5, "test")
    >>> print(result)
    True
    
    Notes
    -----
    Additional implementation notes or mathematical details.
    
    References
    ----------
    .. [1] Author et al. (2020). Title. Journal.
    """
```

### README Updates

When adding features, update:

- Feature list
- Usage examples
- API reference

### CHANGELOG

Follow [Keep a Changelog](https://keepachangelog.com/):

```markdown
## [Unreleased]

### Added
- New shrinkage method `empirical_bayes_v2`
- Support for paired sample designs

### Changed
- Improved performance of chunked processing (2x faster)

### Fixed
- Correct p-value calculation for F-tests with missing data

### Deprecated
- `old_function()` will be removed in v2.0

## [0.1.0] - 2025-01-11

### Added
- Initial release
```

-----

## Performance Optimization

### Benchmarking

Before optimizing:

```python
import time
import cProfile

def benchmark():
    start = time.time()
    result = fit_differential(M, design, ...)
    elapsed = time.time() - start
    print(f"Time: {elapsed:.2f}s")
    print(f"Rate: {len(result)/elapsed:.0f} CpGs/sec")

# Profile hotspots
cProfile.run('benchmark()', sort='cumtime')
```

### Guidelines

1. **Profile first** - Don’t guess at bottlenecks
1. **Vectorize** - Use NumPy operations
1. **Reduce copies** - Modify in-place when safe
1. **Cache results** - Avoid redundant computation
1. **Benchmark** - Measure improvement

**Example optimization**:

```python
# Before (slow)
for i in range(n):
    result[i] = np.exp(array[i]) / (1 + np.exp(array[i]))

# After (fast)
result = np.exp(array) / (1 + np.exp(array))
# or even better:
result = 1 / (1 + np.exp(-array))  # numerically stable
```

-----

## Release Process

For maintainers:

1. Update version in `setup.py`
1. Update [CHANGELOG.md](https://github.com/dare-afolabi/methylation-engine/blob/main/CHANGELOG.md)
1. Create git tag: `git tag v0.0.2`
1. Push tag: `git push origin v0.0.2`
1. Create GitHub release
1. Build and upload to PyPI (if applicable)

-----

## Questions?

- **Email**: [dare-afolabi@outlook.com](dare-afolabi@outlook.com)
- **GitHub Discussions**: Ask questions

-----

## Code of Conduct

Be respectful, inclusive, and professional. See [CODE_OF_CONDUCT.md](https://github.com/dare-afolabi/methylation-engine/blob/main/docs/CODE_OF_CONDUCT.md) for details.

-----

## License

By contributing, you agree that your contributions will be licensed under the same MIT License that covers the project.