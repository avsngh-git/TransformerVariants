"""Root conftest for dashboard tests.

Ensures the repository root is on sys.path so that `from dashboard.components...`
imports work regardless of the working directory used to invoke pytest.
"""

import sys
from pathlib import Path

# Add repository root to sys.path so 'dashboard' package is importable
_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
