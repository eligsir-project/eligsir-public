import json
import os
import tempfile

from scripts.sh_native_backward_diagnostic.assemble_report import assemble_report


def test_assemble_report_includes_all_sections():
    with tempfile.TemporaryDirectory() as tmp:
        synthetic_path = os.path.join(tmp, "synthetic.json")
        audit_path = os.path.join(tmp, "audit.json")
        with open(synthetic_path, "w") as f:
            json.dump({
                "degree_0": {"max_abs_grad_error_dc": 1e-7, "max_abs_grad_error_rest": 1e-7, "max_abs_grad_error_dir": 1e-7},
            }, f)
        with open(audit_path, "w") as f:
            json.dump({
                "means_grad_norm_python": 1.0, "means_grad_norm_gsplat": 1.0001,
                "means_grad_max_abs_diff": 1e-5,
                "sh0_grad_norm_python": 0.5, "sh0_grad_norm_gsplat": 0.5001,
                "shN_grad_norm_python": 0.3, "shN_grad_norm_gsplat": 0.3001,
            }, f)

        report = assemble_report(synthetic_path, audit_path, {"python": "stable, PSNR=20.1", "gsplat": "stable, PSNR=20.0"})

    assert "## Premise re-check" in report
    assert "## Phase B1a" in report
    assert "## Phase B1b" in report
    assert "## Phase B2" in report
    assert "## Recommendation" in report
    assert "stable, PSNR=20.1" in report


def test_assemble_report_includes_confounded_and_deconfounded_numbers():
    with tempfile.TemporaryDirectory() as tmp:
        synthetic_path = os.path.join(tmp, "synthetic.json")
        audit_path = os.path.join(tmp, "audit.json")
        confounded_path = os.path.join(tmp, "confounded.json")
        with open(synthetic_path, "w") as f:
            json.dump({
                "degree_1": {"max_abs_grad_error_dc": 0.0, "max_abs_grad_error_rest": 5.9e-8, "max_abs_grad_error_dir": 1.42},
            }, f)
        with open(audit_path, "w") as f:
            json.dump({
                "means_grad_norm_python": 0.038311831653118134, "means_grad_norm_gsplat": 0.038311831653118134,
                "means_grad_max_abs_diff": 3.176814061589539e-06,
                "sh0_grad_norm_python": 0.002767105819657445, "sh0_grad_norm_gsplat": 0.002767105819657445,
                "shN_grad_norm_python": 0.010716954246163368, "shN_grad_norm_gsplat": 0.010716954246163368,
            }, f)
        with open(confounded_path, "w") as f:
            json.dump({
                "means_grad_norm_python": 0.038169100880622864, "means_grad_norm_gsplat": 0.038311831653118134,
                "means_grad_max_abs_diff": 0.00048170960508286953,
                "sh0_grad_norm_python": 0.0023265888448804617, "sh0_grad_norm_gsplat": 0.0027671060524880886,
                "shN_grad_norm_python": 0.009010840207338333, "shN_grad_norm_gsplat": 0.010716954246163368,
            }, f)

        report = assemble_report(
            synthetic_path, audit_path,
            {"python": "stable, test PSNR=19.15dB", "gsplat": "stable, test PSNR=18.59dB"},
            confounded_gradient_audit_json_path=confounded_path,
        )

    assert "Confounded measurement" in report
    assert "Deconfounded measurement" in report
    assert "0.00048170960508286953" in report or "4.8171e-04" in report.replace(",", "")
    assert "3.176814061589539e-06" in report or "3.1768e-06" in report
