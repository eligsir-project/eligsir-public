"""Regression tests for issue #958.

#794's ``KeyframeAdmitter.precedence_config_warnings`` (populated at
construction by ``validate_backstop_config()``) had no production consumer:
it was read by nothing outside the test suite, never printed on a real run,
and never reached ``report.json``. This module verifies the consumer side
added for #958:

1. ``AdmissionFingerprint`` carries a ``precedence_config_warnings`` field
   and serializes it in ``to_dict()``.
2. ``train_streaming.py`` reads the constructed ``KeyframeAdmitter``'s
   ``precedence_config_warnings`` into the fingerprint near its
   instantiation site.
3. ``train_streaming.py`` prints each warning once at run start, with a
   ``[admission] WARNING (#794 precedence): ...`` prefix -- distinct from
   (not a duplicate of) the print already inside
   ``KeyframeAdmitter.__init__`` itself.

Deliberately does NOT touch ``KeyframeAdmitter``'s construction/validation
code (PR #957, open, owns that file) -- this is consumer-side wiring only,
and torch-free so it runs on a CPU-only CI image.
"""

import ast
from pathlib import Path

from eligsir.utils.admission_fingerprint import AdmissionFingerprint


REPO_ROOT = Path(__file__).parent.parent
TRAIN_STREAMING_PATH = REPO_ROOT / "train_streaming.py"


class TestAdmissionFingerprintPrecedenceWarnings:
    """Behavioral: the dataclass field exists and round-trips through to_dict()."""

    def test_default_is_empty_list(self) -> None:
        fp = AdmissionFingerprint()
        assert fp.precedence_config_warnings == []

    def test_to_dict_includes_precedence_config_warnings(self) -> None:
        fp = AdmissionFingerprint()
        fp.precedence_config_warnings = [
            "max_gap=5 is SHORTER than the velocity-adaptive controller's "
            "nominal target interval"
        ]
        d = fp.to_dict()
        assert "precedence_config_warnings" in d
        assert d["precedence_config_warnings"] == [
            "max_gap=5 is SHORTER than the velocity-adaptive controller's "
            "nominal target interval"
        ]

    def test_to_dict_empty_when_no_warnings(self) -> None:
        fp = AdmissionFingerprint()
        d = fp.to_dict()
        assert d["precedence_config_warnings"] == []

    def test_field_is_a_copy_not_aliased(self) -> None:
        """to_dict() must not alias the live list (mirrors the pattern used
        for admit_owners/gate_bypasses elsewhere in this dataclass)."""
        fp = AdmissionFingerprint()
        fp.precedence_config_warnings = ["w1"]
        d = fp.to_dict()
        d["precedence_config_warnings"].append("w2")
        assert fp.precedence_config_warnings == ["w1"]


class TestTrainStreamingWiring:
    """Static checks that train_streaming.py actually wires the consumer."""

    def test_train_streaming_exists(self) -> None:
        assert TRAIN_STREAMING_PATH.exists()

    def test_fingerprint_populated_from_admitter(self) -> None:
        """The fingerprint's precedence_config_warnings must be set from the
        constructed KeyframeAdmitter's own property, not left at the
        dataclass default forever."""
        content = TRAIN_STREAMING_PATH.read_text()
        assert "_admission_fingerprint.precedence_config_warnings" in content, (
            "train_streaming.py does not populate "
            "_admission_fingerprint.precedence_config_warnings"
        )
        assert "_keyframe_admitter.precedence_config_warnings" in content or (
            "getattr(_keyframe_admitter, \"precedence_config_warnings\"" in content
        ), (
            "train_streaming.py does not read precedence_config_warnings off "
            "the constructed KeyframeAdmitter"
        )

    def test_warnings_printed_at_run_start(self) -> None:
        """Each warning must be printed once, with the documented prefix."""
        content = TRAIN_STREAMING_PATH.read_text()
        assert "[admission] WARNING (#794 precedence):" in content, (
            "train_streaming.py does not print precedence_config_warnings "
            "with the '[admission] WARNING (#794 precedence):' prefix"
        )

    def test_wiring_site_is_valid_python(self) -> None:
        """The whole file must still parse -- guards against a bad edit near
        the fingerprint-init site."""
        content = TRAIN_STREAMING_PATH.read_text()
        ast.parse(content)  # raises SyntaxError on failure

    def test_admitter_access_is_defensive(self) -> None:
        """Per the issue's scope: non-hybrid admission modes (or any future
        mode that skips constructing KeyframeAdmitter) must not raise. The
        wiring line must not do a bare attribute access that would AttributeError
        on a None/absent admitter."""
        content = TRAIN_STREAMING_PATH.read_text()
        assert 'getattr(_keyframe_admitter, "precedence_config_warnings"' in content, (
            "precedence_config_warnings must be read via getattr() with a "
            "default, not a bare attribute access, so a future admission "
            "mode that skips constructing KeyframeAdmitter cannot crash "
            "training"
        )
