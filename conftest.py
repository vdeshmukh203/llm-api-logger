"""Ensure the src/ package layout takes precedence over the top-level .py shim."""
import sys
from pathlib import Path

_src = str(Path(__file__).parent / "src")

# Insert src/ at the very front so the package wins over llm_api_logger.py
if _src not in sys.path:
    sys.path.insert(0, _src)
elif sys.path[0] != _src:
    sys.path.remove(_src)
    sys.path.insert(0, _src)

# Remove any module cached from the legacy .py file before collection starts
for _key in list(sys.modules):
    if _key == "llm_api_logger" or _key.startswith("llm_api_logger."):
        del sys.modules[_key]
