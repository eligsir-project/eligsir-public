"""Admission-fingerprint source guards, grouped by their two consumers.

These retain the existing wiring assertions; token presence is not an execution
or ordering proof. Behavioral admission/report tests remain in the full suite.
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_streaming_fingerprint_wiring():
    source = (ROOT / "train_streaming.py").read_text()
    tree = ast.parse(source)
    imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    imports += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    assert any("admission_fingerprint" in name for name in imports if name)
    required = (
        "_admission_fingerprint = AdmissionFingerprint()",
        "_admission_fingerprint.nominal_policy", "_admission_fingerprint.max_gap",
        "_admission_fingerprint.record_admit", "_admission_fingerprint.record_candidate",
        "_admission_fingerprint.record_reject",
        '_admission_fingerprint.record_reject("tracking_reinit")',
        "_admission_fingerprint.record_reject(decision.reason)",
        "_admission_fingerprint.finalize(", "admission_fingerprint=_admission_fingerprint",
        "streaming/admission/total_admitted", "streaming/admission/total_rejected",
        "streaming/admission/force_hard_count", "streaming/admission/force_soft_count",
    )
    missing = [token for token in required if token not in source]
    assert not missing, f"Missing streaming fingerprint wiring: {missing}"


def test_report_fingerprint_wiring():
    source = (ROOT / "eligsir/utils/comparison_report.py").read_text()
    tree = ast.parse(source)
    functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                 and n.name == "write_post_training_report"]
    assert len(functions) == 1
    assert "admission_fingerprint" in {arg.arg for arg in functions[0].args.kwonlyargs}
    assert 'summary["admission_fingerprint"]' in source
