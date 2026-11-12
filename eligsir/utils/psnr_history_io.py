"""Torch-free reader for ``psnr_history.jsonl``.

Split out of ``utils/psnr_history.py`` (issue #423). That module imports
torch at module scope because its *writer*/tick side is GPU-resident, but
the reader is pure JSONL parsing with no tensor involvement at all -- and
it is what ``utils/baseline_comparison.py`` needs in order to stay
importable in the CPU-only smoke image (``Dockerfile.ci-cpu``, which
deliberately installs no torch).

``utils/psnr_history.py`` re-exports both names, so every existing import
site keeps working unchanged.
"""
from __future__ import annotations

import json
import os

PSNR_REDUCTION_FIELD = "psnr_reduction"
PSNR_REDUCTION_LEGACY_CHW_CHANNEL_MEAN_V1 = "legacy_chw_channel_mean_v1"
PSNR_REDUCTION_RGB_PER_VIEW_V2 = "rgb_per_view_v2"
_PSNR_HISTORY_TAIL_BYTES = 64 * 1024


def psnr_reduction(rows, *, source: str, value_key: str = "mean_test_psnr") -> str | None:
    reductions = {
        str(row.get(PSNR_REDUCTION_FIELD, PSNR_REDUCTION_LEGACY_CHW_CHANNEL_MEAN_V1))
        for row in rows
        if row.get("eval_valid", True) is True and row.get(value_key) is not None
    }
    if len(reductions) > 1:
        raise ValueError(
            f"mixed PSNR reductions in {source}: {', '.join(sorted(reductions))}"
        )
    return next(iter(reductions), None)


def _score_rows(path: str):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def psnr_history_tail_reduction(model_path: str) -> str | None:
    """Return the last scored row's reduction without rereading a curve.

    A legacy curve has an untagged final score and is therefore rejected by
    the writer before a canonical row can be appended. Readers still scan the
    entire file and reject any pre-existing malformed mixed curve.
    """
    path = os.path.join(model_path, "psnr_history.jsonl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        start = max(0, size - _PSNR_HISTORY_TAIL_BYTES)
        f.seek(start)
        if start:
            f.readline()
        lines = f.read().splitlines()
    for line in reversed(lines):
        try:
            row = json.loads(line)
        except (TypeError, ValueError):
            continue
        if (
            isinstance(row, dict)
            and row.get("eval_valid", True) is True
            and row.get("mean_test_psnr") is not None
        ):
            return str(
                row.get(PSNR_REDUCTION_FIELD, PSNR_REDUCTION_LEGACY_CHW_CHANNEL_MEAN_V1)
            )
    if start:
        raise ValueError(
            f"cannot establish PSNR reduction from bounded tail of {path}; "
            "use a fresh model/output path"
        )
    return None


def read_psnr_history(model_path: str):
    """Public reader for another run's psnr_history.jsonl (e.g. a
    --baseline_run comparison target). See _read_psnr_history for the
    same-run internal alias this delegates to."""
    return _read_psnr_history(model_path)


def read_psnr_history_failures(model_path: str):
    """Return durable invalid-tick records without treating them as scores."""
    path = os.path.join(model_path, "psnr_history.jsonl")
    failures = []
    if not os.path.exists(path):
        return failures
    with open(path) as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("eval_valid") is False:
                failures.append(row)
    return failures


def _read_psnr_history(model_path: str):
    path = os.path.join(model_path, "psnr_history.jsonl")
    rows = _score_rows(path)
    psnr_reduction(rows, source=path)
    rows = [
        row
        for row in rows
        if row.get("eval_valid", True) is True and row.get("mean_test_psnr") is not None
    ]
    iters = [row["iteration"] for row in rows]
    vals = [row["mean_test_psnr"] for row in rows]
    return iters, vals
