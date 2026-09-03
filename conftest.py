import sys
from pathlib import Path

# Make the repository root importable as `src.*` regardless of how pytest is
# invoked (`pytest`, `python -m pytest`, or from a subdirectory).
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
