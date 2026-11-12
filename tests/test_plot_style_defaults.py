from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import matplotlib

from eligsir.plot_style import matplotlibrc_path

ROOT = Path(__file__).resolve().parents[1]
# Direct CI-routing references. The root mirror is intentionally absent from the
# lean CPU image; the packaged copy is the runtime contract exercised below.
ROOT_RC = "matplotlibrc"
PACKAGE_RC = "eligsir/matplotlibrc"
PALETTE = (
    "#2385d9",
    "#20b8c5",
    "#9371df",
    "#ed6269",
    "#24ac8d",
    "#e89435",
    "#d9ac24",
    "#da609c",
    "#3486dd",
    "#30b8b2",
)


def _clean_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("MATPLOTLIBRC", None)
    env.pop("ELIGSIR_PLOT_STYLE", None)
    env["PYTHONPATH"] = str(ROOT)
    return env


def test_packaged_rc_uses_canonical_pastel_palette() -> None:
    assert PACKAGE_RC.endswith(matplotlibrc_path().as_posix().split("/eligsir/", 1)[-1])
    assert ROOT_RC == "matplotlibrc"
    rc = matplotlib.rc_params_from_file(
        matplotlibrc_path(), fail_on_error=True, use_default_template=False
    )
    assert tuple(rc["axes.prop_cycle"].by_key()["color"]) == PALETTE
    assert rc["axes.grid"] is True
    assert rc["axes.titleweight"] == "bold"
    assert rc["lines.linewidth"] == 1.8


def test_eligsir_import_enables_style_outside_repo_cwd(tmp_path: Path) -> None:
    code = (
        "import os, matplotlib, eligsir; "
        "print(os.environ['MATPLOTLIBRC']); "
        "print(matplotlib.matplotlib_fname())"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=_clean_env(),
        check=True,
        capture_output=True,
        text=True,
    )
    selected, resolved = result.stdout.strip().splitlines()[-2:]
    assert Path(selected).resolve() == matplotlibrc_path().resolve()
    assert Path(resolved).resolve() == matplotlibrc_path().resolve()


def test_utils_import_enables_style_outside_repo_cwd(tmp_path: Path) -> None:
    code = "import os, utils; print(os.environ['MATPLOTLIBRC'])"
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=_clean_env(),
        check=True,
        capture_output=True,
        text=True,
    )
    assert Path(result.stdout.strip().splitlines()[-1]).resolve() == matplotlibrc_path().resolve()


def test_explicit_opt_out_is_respected(tmp_path: Path) -> None:
    env = _clean_env()
    env["ELIGSIR_PLOT_STYLE"] = "0"
    code = "import os, eligsir; print(os.environ.get('MATPLOTLIBRC', 'unset'))"
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip().splitlines()[-1] == "unset"
