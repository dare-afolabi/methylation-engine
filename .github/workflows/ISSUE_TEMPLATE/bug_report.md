---
name: Bug report
about: Report a problem or unexpected behavior when using methylation-engine as a library
title: ''
labels: bug
assignees: ''

---

**Description**

Provide a clear and concise description of the issue or unexpected behavior observed when using **methylation engine** in Python.

---

**Code Example to Reproduce**

Please include a minimal reproducible snippet:

```bash
import pandas as pd
from core.engine import fit_differential

# Example setup
M = pd.DataFrame(…)
design = pd.DataFrame(…)
contrast = np.array([0.0, 1.0])

# Function call
results = fit_differential(M, design, contrast=contrast)
```

Include any necessary input data structure or shapes to reproduce the issue.

---

**Expected Behavior**

Describe what you expected the function to return or do (e.g., return a DataFrame with columns \[`logFC`, `pval`, …]).

---

**Actual Behavior**

Describe what actually happened (e.g., exception message, NaNs in results, mismatched output shape).

Paste full error traceback here if available.

---

**Environment**

Please provide:
- **OS**: \[e.g. macOS 14, Ubuntu 22.04, Windows 11]
- **EDI/Console**: \[e.g. VS Code]
- **Python version**: \[e.g. 3.11]
- **methylation-engine version**: \[e.g. 0.1.1]
- **Installation method**: \[e.g. `pip install .`, `pip install -e .`]

---

**Additional Context**

Add any other context that might help diagnose the issue:
- Dataset type (e.g. 450k, EPIC)
- Function(s) involved (`fit_differential`, `_add_group_means`, etc.)
- Whether it happens consistently or only under certain data conditions.