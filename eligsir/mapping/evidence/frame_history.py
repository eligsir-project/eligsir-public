"""Per-frame evidence history (plan section 10.4/10.5).

Unlike per-Gaussian evidence (``gaussian_history.py``), per-frame evidence
(``InputEvidence``, ``RenderEvidence``, ``EvidenceSummary``) is naturally
keyed by ``frame_id``, not by a mutable topology-dependent index — a
frame's evidence never becomes ambiguous the way a Gaussian slot does after
a structural mutation. ``FrameEvidenceHistory`` is therefore a plain
bounded ring buffer, not a version-gated store.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Iterable, Optional

from eligsir.mapping.evidence.contracts import (
    EvidenceSummary,
    InputEvidence,
    RenderEvidence,
)


@dataclass
class FrameEvidenceRecord:
    """Single frame's collected evidence (input, render, and derived summary).

    Attributes:
        frame_id: Stable frame identifier from the source trajectory.
        input_evidence: Input-side evidence (image quality, depth statistics, etc.).
        render_evidence: Render-side evidence (loss breakdown, per-view errors, etc.).
        summary: Aggregate summary derived from input and render evidence.
    """

    frame_id: int
    input_evidence: Optional[InputEvidence] = None
    render_evidence: Optional[RenderEvidence] = None
    summary: Optional[EvidenceSummary] = None


@dataclass
class FrameEvidenceHistory:
    """Bounded ring buffer of :class:`FrameEvidenceRecord`, most-recent-last.

    ``max_frames`` bounds memory (32 frames — evidence retention beyond the
    frames the mapper itself still keeps around is of limited use for now)
    — not wired to any CLI flag in this PR since nothing reads it yet.

    Note (issue #360): this docstring used to justify the 32 as "matches
    ``--streaming_replay_buffer``'s default"; that default is actually 512,
    and the replay ring buffer's size is inert at default settings anyway
    (the never-evicted global reservoir takes every global-replay draw).
    The 32 here is simply this class's own independent bound.
    """

    max_frames: int = 32
    _records: Deque[FrameEvidenceRecord] = field(default_factory=deque, repr=False)

    def __post_init__(self) -> None:
        self._records = deque(self._records, maxlen=self.max_frames)

    def append(self, record: FrameEvidenceRecord) -> None:
        """Add a frame evidence record to the buffer (evicting oldest if full).

        Args:
            record: Frame evidence to store.
        """
        self._records.append(record)

    def latest(self) -> Optional[FrameEvidenceRecord]:
        """Return the most recently appended frame evidence record, or None if empty."""
        return self._records[-1] if self._records else None

    def get(self, frame_id: int) -> Optional[FrameEvidenceRecord]:
        """Retrieve evidence for a specific frame by ID, or None if not in buffer.

        Args:
            frame_id: Frame identifier to search for.

        Returns:
            The most recent record matching the frame_id, or None if not found.
        """
        for record in reversed(self._records):
            if record.frame_id == frame_id:
                return record
        return None

    def __iter__(self) -> Iterable[FrameEvidenceRecord]:
        return iter(self._records)

    def __len__(self) -> int:
        return len(self._records)
