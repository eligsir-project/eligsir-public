"""Issue #1074: the candidates layer's buffer ref is NOT the canonical one.

Until 2026-08-23 ``eligsir.mapping.candidates.contracts`` defined a class
literally named ``BufferRef``, colliding with the canonical
``eligsir.contracts.runtime.BufferRef``. That collision made the candidates
type look like a duplicate awaiting migration, when in fact the two express
opposite halves of the decode boundary.

These tests pin the two facts that make unifying them a design change rather
than a rename, so a future migration attempt re-derives the cost instead of
discovering it after the fact:

1. the candidates ref must be able to say "not read yet" / "absent";
2. the canonical ref structurally cannot say either of those things.
"""

from __future__ import annotations

import numpy as np
import pytest

from eligsir.contracts.runtime import BufferOwnership, BufferRef, BufferSpec, MemoryDomain
from eligsir.mapping.candidates import contracts as candidate_contracts
from eligsir.mapping.candidates.contracts import UndecodedBufferRef


def test_candidates_layer_no_longer_exports_the_colliding_name():
    """Exactly one ``BufferRef`` name exists in the codebase."""
    import eligsir.mapping.candidates as candidates_pkg

    assert not hasattr(candidate_contracts, "BufferRef")
    assert not hasattr(candidates_pkg, "BufferRef")
    assert "BufferRef" not in candidates_pkg.__all__
    assert "UndecodedBufferRef" in candidates_pkg.__all__


def test_undecoded_ref_expresses_absent_and_not_yet_read_buffers():
    """The two states the canonical record has no way to represent."""
    absent = UndecodedBufferRef()
    assert absent.is_empty() is True

    not_yet_read = UndecodedBufferRef(path="/frames/000123.png")
    assert not_yet_read.is_empty() is False
    # Deferred by construction: naming the buffer costs no decode.
    assert not_yet_read.array is None
    assert not_yet_read.data is None


def test_canonical_bufferref_cannot_express_an_absent_or_undecoded_buffer():
    """BufferSpec requires a concrete shape/dtype/encoding.

    This is the structural blocker: the canonical record is a payload with
    known layout, so neither ``UndecodedBufferRef()`` (absent) nor
    ``UndecodedBufferRef(path=...)`` (not read yet) has a canonical
    counterpart that could be built without decoding first.
    """
    with pytest.raises(ValueError):
        BufferSpec(
            shape=(),
            dtype="uint8",
            memory_domain=MemoryDomain.HOST,
            encoding="rgb8",
        )
    with pytest.raises(ValueError):
        BufferSpec(
            shape=(8, 8, 3),
            dtype="",
            memory_domain=MemoryDomain.HOST,
            encoding="rgb8",
        )
    with pytest.raises(ValueError):
        BufferSpec(
            shape=(8, 8, 3),
            dtype="uint8",
            memory_domain=MemoryDomain.HOST,
            encoding="",
        )


def test_canonical_bufferref_is_the_right_type_once_pixels_exist():
    """The counterpart to the test above: after decode, canonical applies.

    Documents where a future unification belongs -- at the
    ``preparation.decode_*`` boundary, where real shape/dtype/encoding first
    exist -- not at packet construction.
    """
    decoded = np.zeros((8, 8, 3), dtype=np.uint8)
    ref = BufferRef.from_numpy(decoded, encoding="rgb8")
    assert ref.spec.shape == (8, 8, 3)
    assert ref.spec.dtype == "uint8"
    assert ref.ownership is BufferOwnership.OWNED
