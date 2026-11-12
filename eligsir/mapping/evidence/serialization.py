"""Serialization for evidence logs (plan section 10.4, "Add serialization
and shadow logs").

The goal (per plan section 10.7 deliverable 6): a frozen evidence log
recorded during a real training run can be replayed offline into policy
code (e.g. W3-06's shadow-mode replay) **without rerunning rendering** —
the tensors evidence carries (residuals, alpha, per-Gaussian gradients,
etc.) are exactly what a policy needs, so writing them to disk once and
reading them back is sufficient; no camera/render/backward pass is
required to reconstruct them.

On-disk layout, one directory per log:

```
<log_dir>/
  manifest.jsonl        # one line per record: scalar fields + tensor shard filename
  tensors/
    000000_render.pt     # torch.save'd dict of RenderEvidence tensor fields (frame 0)
    000000_gaussian.pt    # torch.save'd dict of GaussianEvidence tensor fields (frame 0)
    ...
```

Tensors are stored CPU-side (``.cpu()``) so a log recorded on one GPU-backed
run can be replayed on a machine with no GPU at all (pure offline policy
analysis) — matching this project's existing convention of CPU-only-feasible
smoke paths (see ``CLAUDE.md``'s "Smoke test (fast, CPU-only feasible)").
``InputEvidence``/``EvidenceSummary`` are plain floats and go straight into
the JSONL manifest with no separate tensor shard.
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import asdict
from typing import Any, Optional

import torch

from eligsir.mapping.evidence.contracts import (
    EvidenceMeta,
    EvidenceSummary,
    GaussianEvidence,
    InputEvidence,
    RenderEvidence,
)

_RENDER_TENSOR_FIELDS = ("rgb_residual", "depth_residual", "alpha", "rendered_depth", "transmittance")
_GAUSSIAN_TENSOR_FIELDS = (
    "visibility",
    "projected_radius",
    "contribution",
    "xyz_grad_ema",
    "opacity_grad_ema",
    "support",
    "geometry_confidence",
    "free_space_conflict",
    "age",
)


def _meta_dict(meta: EvidenceMeta) -> dict:
    return asdict(meta)


def _meta_from_dict(d: dict) -> EvidenceMeta:
    return EvidenceMeta(**d)


def _tensors_to_cpu_dict(obj: Any, field_names: tuple) -> dict:
    out: dict = {}
    for name in field_names:
        val = getattr(obj, name)
        out[name] = val.detach().cpu() if val is not None else None
    return out


class EvidenceLogWriter:
    """Appends evidence records to an on-disk log directory.

    Each ``write_*`` call is a single, self-contained record (one line in
    ``manifest.jsonl`` plus, for tensor-carrying types, one shard file under
    ``tensors/``). Records are append-only and never rewritten, so a writer
    can be safely shared across a whole training run without holding
    anything in memory beyond the current record.
    """

    def __init__(self, log_dir: str) -> None:
        self.log_dir = log_dir
        self.tensor_dir = os.path.join(log_dir, "tensors")
        os.makedirs(self.tensor_dir, exist_ok=True)
        self._manifest_path = os.path.join(log_dir, "manifest.jsonl")
        self._seq = 0

    def _next_shard_stem(self) -> str:
        stem = f"{self._seq:06d}"
        self._seq += 1
        return stem

    def _append_manifest_line(self, record: dict) -> None:
        with open(self._manifest_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def write_input_evidence(self, evidence: InputEvidence) -> None:
        """Append input evidence (image quality, etc.) to the log.

        Args:
            evidence: InputEvidence to record.
        """
        record = asdict(evidence)
        record["kind"] = "input"
        self._append_manifest_line(record)

    def write_render_evidence(self, evidence: RenderEvidence) -> None:
        """Append render evidence (residuals, depth, etc.) to the log.

        Writes tensor fields to a shard file and metadata to manifest.

        Args:
            evidence: RenderEvidence to record.
        """
        stem = self._next_shard_stem()
        shard_name = f"{stem}_render.pt"
        torch.save(_tensors_to_cpu_dict(evidence, _RENDER_TENSOR_FIELDS), os.path.join(self.tensor_dir, shard_name))
        self._append_manifest_line({"kind": "render", "meta": _meta_dict(evidence.meta), "shard": shard_name})

    def write_gaussian_evidence(self, evidence: GaussianEvidence) -> None:
        """Append per-Gaussian evidence to the log.

        Writes tensor fields to a shard file and metadata to manifest.

        Args:
            evidence: GaussianEvidence to record.
        """
        stem = self._next_shard_stem()
        shard_name = f"{stem}_gaussian.pt"
        torch.save(_tensors_to_cpu_dict(evidence, _GAUSSIAN_TENSOR_FIELDS), os.path.join(self.tensor_dir, shard_name))
        self._append_manifest_line({"kind": "gaussian", "meta": _meta_dict(evidence.meta), "shard": shard_name})

    def write_summary(self, summary: EvidenceSummary) -> None:
        """Append summary evidence to the log.

        Args:
            summary: EvidenceSummary to record.
        """
        record = asdict(summary)
        record["kind"] = "summary"
        self._append_manifest_line(record)


class EvidenceLogReader:
    """Reads back an :class:`EvidenceLogWriter`-written log directory.

    Iterates the manifest and, for tensor-carrying records, loads the
    corresponding shard on demand (lazily — a caller that only wants
    ``InputEvidence``/``EvidenceSummary`` records never touches disk for
    the tensor shards). ``map_location="cpu"`` always, per module docstring.
    """

    def __init__(self, log_dir: str) -> None:
        self.log_dir = log_dir
        self.tensor_dir = os.path.join(log_dir, "tensors")
        self._manifest_path = os.path.join(log_dir, "manifest.jsonl")

    def __iter__(self):
        if not os.path.exists(self._manifest_path):
            return
        with open(self._manifest_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                yield self._materialize(record)

    def _materialize(self, record: dict):
        kind = record["kind"]
        if kind == "input":
            record = dict(record)
            record.pop("kind")
            record["meta"] = _meta_from_dict(record["meta"])
            return InputEvidence(**record)
        if kind == "summary":
            record = dict(record)
            record.pop("kind")
            record["meta"] = _meta_from_dict(record["meta"])
            return EvidenceSummary(**record)
        if kind == "render":
            tensors = torch.load(
                os.path.join(self.tensor_dir, record["shard"]), map_location="cpu", weights_only=True
            )
            return RenderEvidence(meta=_meta_from_dict(record["meta"]), **tensors)
        if kind == "gaussian":
            tensors = torch.load(
                os.path.join(self.tensor_dir, record["shard"]), map_location="cpu", weights_only=True
            )
            return GaussianEvidence(meta=_meta_from_dict(record["meta"]), **tensors)
        raise ValueError(f"Unknown evidence record kind: {kind!r}")

    def read_all(self) -> list:
        """Load all records from the log into memory.

        Returns:
            List of all evidence records (InputEvidence, RenderEvidence,
            GaussianEvidence, EvidenceSummary objects).
        """
        return list(self)
