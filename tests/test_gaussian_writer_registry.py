"""Tests for the issue #325 Gaussian writer/owner registry generator.

Deliberately torch-free (stdlib + pytest only) so these really execute in the
lean CPU CI image -- per #536 a module-scope ``import torch`` silently excludes
a test file from that image, and a green CPU check would then prove nothing
about this guard.

What is and is not enforced here
--------------------------------
ENFORCED: the annotation layer in ``scripts/gaussian_writer_registry.py`` stays
in sync with what the AST scanner actually discovers -- in BOTH directions. A
newly-added Gaussian write with no ownership annotation fails
``test_no_unclassified_mutation_paths``; an annotation whose target write was
deleted or reshaped fails ``test_no_stale_annotations``. That is the drift
guard #325 is really about.

NOT ENFORCED: byte-freshness of a checked-in live scan. The generated
``gaussian_writer_registry.scan.{json,md}`` artifacts embed line numbers, so an
unrelated edit above a write would create false freshness churn. Generate them
for the exact checkout with ``python3 scripts/gaussian_writer_registry.py
--write``; the checked-in vocabulary is intentionally stable and makes no
ownership or sole-writer claims.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import gaussian_writer_discovery as gwd  # noqa: E402
import gaussian_writer_registry as gwr  # noqa: E402


@pytest.fixture(scope="module")
def result():
    return gwr.build_registry()


@pytest.fixture(scope="module")
def conflicts(result):
    return gwr.conflict_report(result)


# ---------------------------------------------------------------------------
# The drift guard (both directions)
# ---------------------------------------------------------------------------


def test_no_unclassified_mutation_paths(result):
    assert result.unclassified == [], (
        "New Gaussian write site(s) with no ownership annotation. Add an entry "
        "to ANNOTATIONS in scripts/gaussian_writer_registry.py (issue #325 "
        "acceptance bar: every production writer is classified):\n  "
        + "\n  ".join(result.unclassified)
    )


def test_no_stale_annotations(result):
    assert result.stale_annotations == [], (
        "Annotation entries that no longer match any discovered write. The "
        "code moved; update or delete the entry:\n  "
        + "\n  ".join(result.stale_annotations)
    )


def test_unclassified_rows_fail_closed_on_unowned_production_mutation():
    assert gwr._unclassified_rows(
        [
            _row(
                semantic_operation="strategy_dispatch",
                declared_owner="none",
                linked_to_transaction=False,
            )
        ]
    ) == ["a/b.py::f"]


def test_unclassified_rows_fail_closed_on_unowned_parameter_write():
    assert gwr._unclassified_rows(
        [
            _row(
                state_written=["opacity"],
                semantic_operation="opacity",
                declared_owner="none",
                linked_to_transaction=False,
            )
        ]
    ) == ["a/b.py::f"]


def test_unclassified_rows_ignore_nonproduction_diagnostics():
    assert gwr._unclassified_rows(
        [
            _row(
                path="scripts/diagnostic.py",
                semantic_operation="strategy_dispatch",
                declared_owner="none",
                linked_to_transaction=False,
                production=False,
            )
        ]
    ) == []


def test_production_runtime_mutation_is_owned_by_issue1193_barrier(result):
    key = (
        "eligsir/runtime/production_application.py::"
        "CanonicalMappingRuntime._mutate_after_step"
    )
    rows = [row for row in result.rows if row.key == key]
    assert rows == []
    assert key not in result.unclassified
    source = (REPO_ROOT / "eligsir/runtime/production_application.py").read_text()
    assert "GaussianMutationEpoch(" in source
    assert "epoch.commit(proposals)" in source


def test_controlled_vocabularies_are_respected(result):
    assert result.vocabulary_errors == []
    for row in result.rows:
        assert row.semantic_operation in gwr.SEMANTIC_OPERATIONS
        assert row.declared_owner in gwr.OWNERS


def test_registry_is_non_trivial(result):
    # A scanner that silently stopped matching would otherwise pass every
    # assertion above by finding nothing at all.
    assert result.n_sites > 300
    assert result.n_units > 100
    assert len(result.rows) >= result.n_units


def test_scan_roots_all_exist():
    for root in gwd.SCAN_ROOTS:
        assert (REPO_ROOT / root).exists(), (
            f"SCAN_ROOTS entry {root!r} does not exist. It is kept in lockstep "
            "with the COPY lines in Dockerfile.ci-cpu."
        )


# ---------------------------------------------------------------------------
# Scanner behaviour (synthetic sources -- no repo coupling)
# ---------------------------------------------------------------------------


def _scan_source(tmp_path: Path, src: str):
    f = tmp_path / "mod.py"
    f.write_text(src, encoding="utf-8")
    return gwd.discover(tmp_path, roots=("mod.py",))


def test_scanner_finds_raw_param_writes(tmp_path):
    sites = _scan_source(
        tmp_path,
        "def f(g):\n"
        "    g.params['means'].data[m] = x\n"
        "    g.params['opacities'].data[m] = -9.21\n",
    )
    groups = {s.state_group for s in sites}
    assert groups == {"means_xyz", "opacity"}


def test_scanner_follows_function_local_aliases(tmp_path):
    sites = _scan_source(
        tmp_path,
        "def f(g):\n"
        "    op = g.params['opacities'].data\n"
        "    op[mask] = 0.0\n",
    )
    assert [s.state_group for s in sites] == ["opacity"]
    assert sites[0].kind == "alias_assign"


def test_scanner_finds_inplace_mutators(tmp_path):
    sites = _scan_source(tmp_path, "def f(g):\n    g._xyz.add_(noise)\n")
    assert [(s.state_group, s.kind) for s in sites] == [("means_xyz", "inplace")]


def test_scanner_finds_setattr_of_shadow_buffers(tmp_path):
    sites = _scan_source(tmp_path, "def f(g):\n    setattr(g, 'birth_frame', z)\n")
    assert [(s.state_group, s.kind) for s in sites] == [("lifecycle_state", "setattr")]


def test_scanner_ignores_bare_local_names(tmp_path):
    # `provisional = gaussians.provisional` is a READ binding, not a write.
    sites = _scan_source(tmp_path, "def f(g):\n    provisional = g.provisional\n")
    assert sites == []


def test_attribute_match_is_identifier_bounded(tmp_path):
    # `_xyz_grad_ema` is an evidence EMA buffer, not the model's `_xyz`.
    sites = _scan_source(tmp_path, "def f(self):\n    self._xyz_grad_ema = e\n")
    assert sites == []


def test_scanner_records_enclosing_qualname(tmp_path):
    sites = _scan_source(
        tmp_path,
        "class M:\n    def prune(self, m):\n        self.params['scales'].data[m] = 0\n",
    )
    assert sites[0].qualname == "M.prune"


def test_scanner_is_stdlib_only():
    # Guards the property that makes these tests run in the CPU CI image.
    for mod in (gwd, gwr):
        tree = ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert "torch" not in imported
        assert "numpy" not in imported


# ---------------------------------------------------------------------------
# Conflict checks, on hand-built rows (independent of the live repo state)
# ---------------------------------------------------------------------------


def _row(**kw):
    base = dict(
        path="a/b.py",
        qualname="f",
        state_written=["population"],
        semantic_operation="grow",
        declared_owner="mcmc",
        trigger="t",
        gate="g",
        evidence_consumed="e",
        hard_safety=False,
        production=True,
        is_mechanism=False,
        sanctioned_raw_access=False,
        coordinate_correction=False,
        updates_anchor_reference=False,
        linked_to_transaction=True,
        notes="",
        lines=[1],
        kinds=["model_method"],
        examples=[],
    )
    base.update(kw)
    return gwr.RegistryRow(**base)


def _report(rows):
    return gwr.conflict_report(
        gwr.RegistryResult(
            rows=rows, unclassified=[], stale_annotations=[], vocabulary_errors=[],
            n_sites=len(rows), n_units=len(rows),
        )
    )


def test_check_a_flags_two_non_safety_owners():
    rep = _report([_row(declared_owner="mcmc"), _row(path="c/d.py", declared_owner="source_depth")])
    assert rep["check_a_duplicate_non_safety_owners"]["grow"]["n_owners"] == 2


def test_check_a_ignores_hard_safety_and_mechanism_owners():
    rep = _report(
        [
            _row(declared_owner="mcmc"),
            _row(path="c/d.py", declared_owner="freespace_contradiction", hard_safety=True),
            _row(path="e/f.py", declared_owner="mutation_transaction", is_mechanism=True),
        ]
    )
    assert rep["check_a_duplicate_non_safety_owners"] == {}


def test_check_a_ignores_non_production_rows():
    rep = _report(
        [_row(declared_owner="mcmc"), _row(path="scripts/x.py", declared_owner="none", production=False)]
    )
    assert rep["check_a_duplicate_non_safety_owners"] == {}


def test_check_b_flags_raw_writes_outside_sanctioned_layers():
    rep = _report(
        [
            _row(state_written=["opacity"], kinds=["assign"], sanctioned_raw_access=False),
            _row(
                path="scene/gsplat_model.py",
                state_written=["opacity"],
                kinds=["assign"],
                sanctioned_raw_access=True,
            ),
        ]
    )
    flagged = rep["check_b_raw_writes_outside_sanctioned_layers"]
    assert [f["unit"] for f in flagged] == ["a/b.py::f"]


def test_check_c_flags_unlinked_topology_mutation():
    rep = _report(
        [
            _row(linked_to_transaction=False),
            _row(path="c/d.py", linked_to_transaction=True),
        ]
    )
    flagged = rep["check_c_topology_mutation_not_transaction_linked"]
    assert [f["unit"] for f in flagged] == ["a/b.py::f"]


def test_check_c_separates_capacity_only_mutations_from_topology():
    # An opacity floor takes a Gaussian out of service without changing row
    # count -- it must NOT be counted as a topology mutation.
    rep = _report(
        [
            _row(
                path="cap/x.py",
                semantic_operation="retire",
                state_written=["opacity"],
                kinds=["assign"],
                linked_to_transaction=False,
            )
        ]
    )
    assert rep["check_c_topology_mutation_not_transaction_linked"] == []
    assert [f["unit"] for f in rep["check_c_capacity_mutation_not_transaction_linked"]] == [
        "cap/x.py::f"
    ]


def test_check_a_sensitivity_reports_what_flipping_hard_safety_would_do():
    rep = _report(
        [
            _row(declared_owner="mcmc"),
            _row(path="c/d.py", declared_owner="freespace_contradiction", hard_safety=True),
        ]
    )
    assert rep["check_a_duplicate_non_safety_owners"] == {}
    sens = rep["check_a_hard_safety_sensitivity"]
    assert [s["unit"] for s in sens] == ["c/d.py::f"]
    assert sens[0]["check_a_operations_if_not_hard_safety"] == ["grow"]


def test_check_d_breakdown_separates_call_sites_from_shared_primitives():
    rep = _report(
        [
            _row(
                path="loop_closure/x.py",
                semantic_operation="coordinate_correction",
                coordinate_correction=True,
            ),
            _row(
                path="eligsir/map/mutation.py",
                semantic_operation="coordinate_correction",
                coordinate_correction=True,
                is_mechanism=True,
            ),
        ]
    )
    bd = rep["check_d_breakdown"]
    assert bd["production_correction_call_sites"] == ["loop_closure/x.py::f"]
    assert bd["shared_write_primitives"] == ["eligsir/map/mutation.py::f"]


def test_check_d_flags_coordinate_correction_without_anchor_update():
    rep = _report(
        [
            _row(
                semantic_operation="coordinate_correction",
                coordinate_correction=True,
                updates_anchor_reference=False,
            ),
            _row(
                path="c/d.py",
                semantic_operation="coordinate_correction",
                coordinate_correction=True,
                updates_anchor_reference=True,
            ),
        ]
    )
    flagged = rep["check_d_coordinate_correction_without_anchor_update"]
    assert [f["unit"] for f in flagged] == ["a/b.py::f"]


# ---------------------------------------------------------------------------
# Findings that must not silently disappear (regression anchors)
# ---------------------------------------------------------------------------


def test_known_coordinate_correction_sites_are_still_discovered(result):
    keys = {r.key for r in result.rows}
    for expected in (
        "loop_closure/submap_correction.py::apply_submap_correction",
        "utils/submap_manager.py::SubmapManager.apply_pose_graph_corrections",
        "eligsir/map/mutation.py::GsplatMutationTransaction.apply_pose_correction",
    ):
        assert expected in keys


def test_conflict_report_shape(conflicts):
    for key in (
        "check_a_duplicate_non_safety_owners",
        "check_a_family_duplicate_non_safety_owners",
        "check_b_raw_writes_outside_sanctioned_layers",
        "check_a_hard_safety_sensitivity",
        "check_c_topology_mutation_not_transaction_linked",
        "check_c_capacity_mutation_not_transaction_linked",
        "check_d_coordinate_correction_without_anchor_update",
        "check_d_breakdown",
    ):
        assert key in conflicts


def test_summary_reports_the_headline_numbers(result, conflicts):
    s = gwr.summary(result, conflicts)
    assert s["no_unclassified_mutation_paths"] is True
    assert s["n_unclassified_sites"] == 0
    assert isinstance(s["n_semantic_operations_with_competing_non_safety_owners"], int)


def test_payload_is_json_serialisable(result, conflicts):
    import json

    payload = gwr.to_payload(result, conflicts)
    gwr._fill_competitors(payload)
    json.dumps(payload)
    assert payload["registry"]
    first = payload["registry"][0]
    for column in (
        "file/function",
        "state_written",
        "semantic_operation",
        "declared_owner",
        "trigger/cadence",
        "gate",
        "evidence_consumed",
        "hard_safety?",
        "potential_competitors",
        "production_reader/writer test",
    ):
        assert column in first, f"required #325 column {column!r} missing"


def test_write_keeps_generated_scan_separate_from_vocabulary_contract(
    tmp_path, monkeypatch, result, conflicts
):
    import json

    payload = gwr.to_payload(result, conflicts)
    gwr._fill_competitors(payload)
    monkeypatch.setattr(gwr, "REPO_ROOT", tmp_path)

    gwr._write(payload)

    out_dir = tmp_path / "docs" / "architecture"
    assert json.loads(
        (out_dir / "gaussian_writer_registry.scan.json").read_text(encoding="utf-8")
    ) == payload
    assert (out_dir / "gaussian_writer_registry.scan.md").read_text(
        encoding="utf-8"
    ) == gwr.render_markdown(payload)
    assert not (out_dir / "gaussian_writer_vocabulary.json").exists()


def test_markdown_renders(result, conflicts):
    payload = gwr.to_payload(result, conflicts)
    gwr._fill_competitors(payload)
    md = gwr.render_markdown(payload)
    # The artifact lives under docs/architecture/, which scripts/docs_visuals.py
    # treats as visual-critical: a changed doc there must carry a visual or
    # declare an exemption. Since this file is machine-generated, the generator
    # emits that frontmatter itself (hand-adding it would be erased by the next
    # --write), so the heading is no longer the first line.
    assert md.startswith("---\nvisual_exempt:")
    assert "\n# Gaussian writer/owner registry" in md
    assert "## Registry" in md
