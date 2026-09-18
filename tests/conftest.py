"""Make `scripts.*` importable from any working directory (paper scripts are tested in place)."""

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
