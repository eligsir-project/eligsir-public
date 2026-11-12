"""Repository-owned plotting defaults.

The style is intentionally presentation-only: plot owners keep their existing
samples, clocks, axes, units, missing-data semantics, colormaps and provenance.
Importing :mod:`eligsir` selects the package-owned ``matplotlibrc`` as the
Matplotlib fallback configuration unless the caller explicitly opted out or
provided its own ``MATPLOTLIBRC``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PLOT_STYLE_ENV = "ELIGSIR_PLOT_STYLE"
MATPLOTLIBRC_ENV = "MATPLOTLIBRC"
_DISABLED = {"0", "false", "no", "off"}


def matplotlibrc_path() -> Path:
    """Return the canonical Matplotlib configuration shipped with EliGSiR."""
    return Path(__file__).resolve().with_name("matplotlibrc")


def svg_output_path(path: str | os.PathLike[str]) -> Path:
    """Return the canonical vector-artifact path for a generated plot."""
    return Path(path).with_suffix(".svg")


def install_default_plot_style() -> bool:
    """Install EliGSiR's plotting defaults for later Matplotlib imports.

    Returns ``True`` only when this call selected the package rc file. Explicit
    caller configuration always wins: ``ELIGSIR_PLOT_STYLE=0`` disables the
    default and an existing ``MATPLOTLIBRC`` is never replaced. If Matplotlib
    was imported unusually early, reload the rc file immediately so package
    imports still converge on the same default presentation.
    """
    if os.environ.get(PLOT_STYLE_ENV, "1").strip().lower() in _DISABLED:
        return False
    if MATPLOTLIBRC_ENV in os.environ:
        return False

    rc_path = matplotlibrc_path()
    os.environ[MATPLOTLIBRC_ENV] = str(rc_path)

    if "matplotlib" in sys.modules:
        import matplotlib

        matplotlib.rc_file(rc_path)
    return True


__all__ = ["install_default_plot_style", "matplotlibrc_path", "svg_output_path"]
