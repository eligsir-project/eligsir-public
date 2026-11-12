"""Issue #279 (Phase 1) -- CPU<->GPU transfer/synchronization contracts.

Two independent things are pinned here:

1. **The guard-resolution scanner itself** (``scripts/audit_hot_loop_guards.py``)
   -- it is the evidence behind the audit's central Phase 1 claim, so it must
   keep locating the real streaming loop and keep producing a fully-resolved
   guard chain for every site. Pure-AST, no torch needed.

2. **The two host-readback reductions landed for this issue.** These live deep
   inside ``streaming_training()``'s ~6000-line loop body and cannot be called
   directly, so they are pinned the only two honest ways available:
   structurally (the duplicate is gone from the source, and no *unconditional*
   readback exists in the loop body), and semantically (the replacement
   expression for the lifecycle histogram is value-identical to the four
   separate casts it replaces, checked against real tensors).
"""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCANNER_PATH = os.path.join(REPO_ROOT, "scripts", "audit_hot_loop_guards.py")
TRAIN_STREAMING = os.path.join(REPO_ROOT, "train_streaming.py")


def _load_scanner():
    spec = importlib.util.spec_from_file_location("audit_hot_loop_guards", SCANNER_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["audit_hot_loop_guards"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def scan_result():
    return _load_scanner().scan(TRAIN_STREAMING)


# --------------------------------------------------------------------------
# 1. scanner contracts
# --------------------------------------------------------------------------

def test_scanner_locates_the_real_streaming_loop(scan_result):
    """The whole audit rests on this loop being the per-iteration one."""
    assert scan_result["function"] == "streaming_training"
    assert scan_result["loop_line"] > 0
    assert scan_result["loop_end_line"] > scan_result["loop_line"]
    # The streaming loop body is thousands of lines; a few-line match would
    # mean the finder latched onto some other `for iteration in ...`.
    assert scan_result["loop_end_line"] - scan_result["loop_line"] > 1000


def test_every_site_is_fully_classified(scan_result):
    """No site may fall outside the four published verdicts.

    Phase 0 deliberately published an ``UNCLASSIFIED`` bucket. This scanner
    makes the stronger claim that within the loop *body* every site resolves,
    so a new unresolvable verdict is a regression in the evidence.
    """
    allowed = {"UNCONDITIONAL", "CADENCE_GATED", "FEATURE_GATED", "DATA_GATED"}
    assert set(scan_result["by_verdict"]) <= allowed
    assert sum(scan_result["by_verdict"].values()) == scan_result["n_sites"]
    for site in scan_result["sites"]:
        assert site["verdict"] in allowed
        assert site["direction"] in {"D2H", "H2D", "SYNC"}
        # Anything not UNCONDITIONAL must carry the guards that justify it.
        if site["verdict"] != "UNCONDITIONAL":
            assert site["guards"], f"{site['line']} classified gated with no guard chain"


def test_no_unconditional_host_readback_in_the_loop_body(scan_result):
    """The Phase 1 headline result, pinned as a regression guard.

    Every transfer/readback written directly in the streaming loop body sits
    behind a cadence, feature or data guard -- none runs unconditionally on
    every iteration. A new unguarded ``.item()``/``.cpu()``/``.to("cuda")``
    in the loop body would put a fresh implicit device sync on the critical
    path of every single iteration, so it must be a deliberate, reviewed act.
    """
    unconditional = [s for s in scan_result["sites"] if s["verdict"] == "UNCONDITIONAL"]
    assert unconditional == [], (
        "new unconditional per-iteration CPU<->GPU site(s) in the streaming loop: "
        + ", ".join(f"train_streaming.py:{s['line']} {s['label']} {s['source'][:70]}"
                    for s in unconditional)
    )


# --------------------------------------------------------------------------
# 2. the landed reductions
# --------------------------------------------------------------------------

def test_psnr_is_not_recomputed_for_the_viewer(scan_result):
    """The TB block and the viewer block must not each build their own PSNR.

    Both are gated on the same ``iteration % scalar_log_interval == 0``
    cadence over the same ``image``/``gt_image``, so a second call is a second
    PSNR kernel plus a second implicit sync for a value already in hand. The
    viewer's ``_metric_psnr is None`` fallback remains, but applies the same
    declared score transform and only executes without a TensorBoard writer.
    """
    with open(TRAIN_STREAMING, "r", encoding="utf-8") as fh:
        lines = fh.readlines()
        body = "".join(lines)
    # The raw training render must never be scored directly. One declared
    # clamp-pair site serves TB + viewer; a second exists only for viewer-only
    # runs where the shared value is absent.
    raw_psnr_sites = [
        line for line in lines
        if "psnr(image, gt_image)" in line and not line.lstrip().startswith("#")
    ]
    assert raw_psnr_sites == []
    assert body.count("apply_declared_score_clamp_pair(") >= 2
    assert "_metric_psnr = psnr(_score_image, _score_gt).mean().item()" in body
    assert "_metric_psnr if _metric_psnr is not None" in body


def test_shared_metric_values_are_always_defined_before_use(scan_result):
    """`_metric_psnr`/`_metric_l1` must be initialised unconditionally.

    They are read from the viewer block and the controller-pressure block; if
    the initialisation ever moved under a guard, those reads would raise
    ``UnboundLocalError`` on the paths where the guard is false.
    """
    with open(TRAIN_STREAMING, "r", encoding="utf-8") as fh:
        lines = fh.readlines()
    init_psnr = [i for i, l in enumerate(lines) if l.strip() == "_metric_psnr = None"]
    init_l1 = [i for i, l in enumerate(lines) if l.strip() == "_metric_l1 = None"]
    assert len(init_psnr) == 1 and len(init_l1) == 1
    # The unconditional initialisation must precede every consumer.
    first_use = min(
        i for i, l in enumerate(lines)
        if ("_metric_psnr if" in l) or ("_metric_l1 if" in l)
    )
    assert init_psnr[0] < first_use and init_l1[0] < first_use


def test_shared_metric_initialisation_is_unconditional_in_the_loop_body():
    """AST proof that the shared values can never be read unbound.

    ``test_shared_metric_values_are_always_defined_before_use`` only checks
    textual line ordering, which does NOT prove reachability: an
    initialisation that precedes a consumer in the file can still be skipped
    at runtime if it sits under a guard the consumer does not share. This test
    walks the AST of the streaming loop body and asserts the initialisation of
    both shared values has an EMPTY conditional-guard chain -- i.e. it runs on
    every iteration that reaches the logging section, so every consumer
    downstream is guaranteed a bound name.
    """
    import ast

    with open(TRAIN_STREAMING, "r", encoding="utf-8") as fh:
        tree = ast.parse(fh.read())

    scanner = _load_scanner()
    finder = scanner._LoopFinder("streaming_training")
    finder.visit(tree)
    loop = finder.loop
    assert loop is not None

    found: dict[str, int] = {}

    def walk(node, depth: int) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            return
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            if node.value.value is None:
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name) and tgt.id in ("_metric_psnr", "_metric_l1"):
                        found[tgt.id] = depth
        for child in ast.iter_child_nodes(node):
            if isinstance(node, ast.If) and child is not node.test:
                walk(child, depth + 1)
            elif isinstance(node, ast.Try) and child in node.handlers:
                walk(child, depth + 1)
            elif isinstance(node, (ast.For, ast.While)) and child in node.body:
                walk(child, depth + 1)
            else:
                walk(child, depth)

    for stmt in loop.body:
        walk(stmt, 0)

    assert set(found) == {"_metric_psnr", "_metric_l1"}, (
        f"shared-metric initialisation not found in the loop body: {found}"
    )
    for name, depth in found.items():
        assert depth == 0, (
            f"{name} is initialised under {depth} conditional guard(s); it must be "
            "unconditional in the loop body or its consumers can raise UnboundLocalError"
        )


def test_lifecycle_histogram_uses_a_single_readback():
    with open(TRAIN_STREAMING, "r", encoding="utf-8") as fh:
        body = fh.read()
    assert "_lc_counts = torch.stack(" in body
    for stale in (
        'int((lc == 0).sum())',
        'int((lc == 1).sum())',
        'int((lc == 2).sum())',
        'int((lc == 3).sum())',
    ):
        assert stale not in body, f"un-batched lifecycle readback still present: {stale}"


@pytest.mark.parametrize(
    "values",
    [
        [0, 1, 2, 3],
        [0, 0, 0, 0],
        [3, 3, 2, 1, 0, 0],
        [],
        [2] * 257,
        [0, 1, 2, 3, 7],      # out-of-range state must not shift the four buckets
        [-1, 0, 1, 2, 3],     # negative state must not raise (a bincount would)
    ],
)
def test_stacked_lifecycle_counts_match_the_four_separate_casts(values):
    """Semantic parity for the batched histogram readback.

    This is the exact expression substituted into the training loop, checked
    against the exact expression it replaced. The two edge cases at the end are
    why the implementation stacks comparison-sums rather than calling
    ``torch.bincount``: bincount would either raise on a negative value or
    silently re-index, while the four comparisons cannot.
    """
    torch = pytest.importorskip("torch")
    lc = torch.tensor(values, dtype=torch.int64)

    before = [int((lc == s).sum()) for s in (0, 1, 2, 3)]
    after = [int(v) for v in torch.stack([(lc == s).sum() for s in (0, 1, 2, 3)]).tolist()]

    assert after == before
    assert sum(before) <= lc.numel()
