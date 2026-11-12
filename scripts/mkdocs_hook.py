from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.docs_catalog import generate_all  # noqa: E402


def on_config(config):
    """Generate ignored documentation artifacts before MkDocs collects files."""
    generate_all(ROOT)
    return config
