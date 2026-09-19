import sys
from pathlib import Path

# Make `digitood` importable without installing the package, so `pytest` works
# straight after a clone.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
