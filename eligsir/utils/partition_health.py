"""Detect statistically under-reconstructed scene partitions.

A fixed absolute PSNR threshold ("bad" = below 15dB) doesn't generalize --
a hard scene's healthy partitions might all sit around 13dB, while an easy
scene's worst partition might still be 22dB and genuinely fine. This module
instead flags partitions that are *outliers relative to the rest of the
same scene* using robust statistics (median + MAD), so "under-reconstructed"
is scene-relative, not an arbitrary global cutoff.
"""
from __future__ import annotations

from typing import Dict, Set

import numpy as np


def detect_outlier_partitions(
    partition_psnr: Dict[str, float],
    k: float = 1.5,
    min_partitions: int = 3,
    partition_n: "Dict[str, int] | None" = None,
    min_views_per_partition: int = 1,
    direction: str = "low",
) -> Set[int]:
    """Return the set of partition ids whose PSNR is a low outlier.

    Uses the median and MAD (median absolute deviation, scaled by 1.4826
    to be a consistent estimator of the standard deviation under
    normality) across all partitions with data. A partition is flagged
    when its PSNR falls below ``median - k * MAD``.

    Args:
        partition_psnr: e.g. ``{"p0": 21.5, "p1": 12.3, ...}`` (``"--"``/
            None entries for partitions with no test data are ignored).
        k: outlier sensitivity -- lower flags more partitions, higher
            flags fewer. 1.5 is a common robust-stats default (roughly
            analogous to the 1.5*IQR rule of thumb).
        min_partitions: refuse to flag anything when fewer than this many
            partitions have data -- with only 1-2 data points "outlier"
            is not a meaningful concept (and MAD may be 0, which would
            otherwise flag any partition below the median at all).
        partition_n: optional ``{"p0": n_test_views, ...}`` companion dict
            (as produced by ``compute_partition_stats``). When given
            together with ``min_views_per_partition`` > 1, partitions
            backed by fewer test views than that are excluded from both
            the outlier computation and the flagged set -- with finer
            partitioning (see ``partition_count``), a single bad render
            in a 1-view partition is much less confident evidence than
            the same PSNR in a 5-view partition.
        min_views_per_partition: minimum test-view count (see
            ``partition_n``) for a partition to be eligible at all.
            Ignored when ``partition_n`` is not provided.
        direction: ``"low"`` (default) flags under-reconstructed partitions
            (``psnr < median - k*MAD``) -- the original use case. ``"high"``
            flags the mirror case, partitions that are doing unusually
            *well* relative to the rest of the scene (``psnr > median +
            k*MAD``) -- used to identify partitions safe to demote frames
            out of the active training buffer (see StreamingScene.
            demote_partitions), freeing attention for the low outliers.

    Returns:
        Set of partition ids (as ints, parsed from keys like "p3" -> 3)
        that are outliers in the requested direction. Empty set if nothing
        qualifies.
    """
    values = {}
    for key, v in partition_psnr.items():
        if v is None or (isinstance(v, str) and not _is_number(v)):
            continue
        if partition_n is not None and min_views_per_partition > 1:
            n = partition_n.get(key)
            if n is None or int(n) < min_views_per_partition:
                continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        pid = _parse_partition_id(key)
        if pid is not None:
            values[pid] = fv

    if len(values) < min_partitions:
        return set()

    arr = np.array(list(values.values()), dtype=np.float64)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median))) * 1.4826
    if mad <= 1e-9:
        # No spread at all (e.g. every partition is identical) -- nothing
        # is a meaningful outlier regardless of k.
        return set()

    if direction == "high":
        threshold = median + k * mad
        return {pid for pid, v in values.items() if v > threshold}
    threshold = median - k * mad
    return {pid for pid, v in values.items() if v < threshold}


def _parse_partition_id(key) -> "int | None":
    if isinstance(key, int):
        return key
    s = str(key)
    if s.startswith("p") and s[1:].isdigit():
        return int(s[1:])
    if s.isdigit():
        return int(s)
    return None


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False
