#!/usr/bin/env bash
# compare_baseline.sh -- run the test suite against two git trees and diff the
# failure sets, without the silent self-comparison trap of issue #1106.
#
# WHY THIS SCRIPT EXISTS
# ----------------------
# `docker-compose.yml`'s `train` service binds `./:/workspace/eligsir` and
# sets that as `working_dir`. A hand-rolled `-v "$PWD":/workspace/3dgs-mcmc`
# (the recipe CLAUDE.md documented until #1106) mounts to a path nothing reads:
# the container still runs whatever `./` resolves to. A "baseline vs branch"
# comparison written that way silently runs the BRANCH TWICE and reports a
# byte-identical, perfectly green "no regressions" result. That really happened
# while verifying PR #1105, and it hid a genuine regression.
#
# Three properties make this script safe where the hand-rolled flags were not:
#
#   1. SYMMETRY. Both sides are exported with `git archive` and mounted at the
#      same in-container path (/tree). Neither side rides the compose bind.
#      The #1106 bug is structurally a consequence of an asymmetric comparison
#      (one arm via `-v`, the other via `./`); symmetry removes the class.
#   2. REFUSAL. Both refs are resolved to full SHAs up front and the script
#      exits non-zero if they are equal. Documentation cannot do this.
#   3. IDENTITY. Each export carries a TREE_IDENTITY file, printed by
#      tests/conftest.py's `pytest_configure` at the top of every run. Two
#      identical banners side by side is an immediately visible red flag
#      rather than a silent trap; the check below turns that into a refusal.
#
# USAGE
#   scripts/compare_baseline.sh <baseline-ref> [current-ref] [-- <pytest args>]
#
# EXAMPLES
#   scripts/compare_baseline.sh 5fb01cfa
#   scripts/compare_baseline.sh origin/main HEAD
#   scripts/compare_baseline.sh v0.9.0 HEAD -- tests/streaming -x
#
# OUTPUT
#   A work directory (printed at the end, kept for inspection) containing
#   per-side pytest logs, sorted FAILED/ERROR sets, collected-nodeid sets, and
#   the partitioned diff. Exit code 1 if any real regression is found.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTEST_ARGS=(tests/)
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  if [[ "$1" == "--" ]]; then
    shift
    PYTEST_ARGS=("$@")
    break
  fi
  POSITIONAL+=("$1")
  shift
done

if [[ ${#POSITIONAL[@]} -lt 1 ]]; then
  echo "usage: scripts/compare_baseline.sh <baseline-ref> [current-ref] [-- <pytest args>]" >&2
  exit 2
fi

BASELINE_REF="${POSITIONAL[0]}"
CURRENT_REF="${POSITIONAL[1]:-HEAD}"

BASELINE_SHA="$(git rev-parse --verify "${BASELINE_REF}^{commit}")"
CURRENT_SHA="$(git rev-parse --verify "${CURRENT_REF}^{commit}")"

# --- Property 2: refuse a self-comparison -----------------------------------
if [[ "$BASELINE_SHA" == "$CURRENT_SHA" ]]; then
  cat >&2 <<EOF
REFUSED: baseline and current resolve to the same commit.

  baseline '$BASELINE_REF' -> $BASELINE_SHA
  current  '$CURRENT_REF' -> $CURRENT_SHA

Comparing a tree against itself always reports "no regressions". That is the
exact failure mode of issue #1106 and it is never a useful verification.
EOF
  exit 2
fi

WORK="$(mktemp -d "${TMPDIR:-/tmp}/compare_baseline.XXXXXXXX")"
echo "[compare_baseline] work dir: $WORK"
echo "[compare_baseline] baseline '$BASELINE_REF' -> $BASELINE_SHA"
echo "[compare_baseline] current  '$CURRENT_REF' -> $CURRENT_SHA"
echo "[compare_baseline] pytest args: ${PYTEST_ARGS[*]}"

# --- Property 1 + 3: symmetric export, each carrying its own identity --------
export_tree() {
  local sha="$1" dest="$2"
  mkdir -p "$dest"
  git archive "$sha" | tar -x -C "$dest"
  # `git archive` strips .git, so `git rev-parse` cannot work inside the
  # export. Stamp the identity into the tree instead; conftest reads this.
  printf '%s\n' "$sha" >"$dest/TREE_IDENTITY"
}

export_tree "$BASELINE_SHA" "$WORK/baseline"
export_tree "$CURRENT_SHA" "$WORK/current"

# The two exports must not be identical in content either -- different SHAs
# with an identical tree (e.g. a docs-only commit range) still yields a
# vacuous comparison, so say so loudly rather than reporting a fake pass.
if [[ "$(git rev-parse "${BASELINE_SHA}^{tree}")" == "$(git rev-parse "${CURRENT_SHA}^{tree}")" ]]; then
  echo "[compare_baseline] WARNING: the two commits have identical trees; this comparison is vacuous." >&2
fi

run_side() {
  local name="$1" dir="$2" sha="$3"
  local pytest_status=0 collect_status=0
  echo "[compare_baseline] === running $name ($sha) ==="
  # Mount at /tree and `cd` there. Note this does NOT unmount the compose
  # bind at /workspace/eligsir -- the current tree is still present in the
  # container. What makes the run honest is that cwd is /tree and pyproject's
  # `pythonpath = ["."]` plus python's cwd-first sys.path resolution import
  # from there. The tree-identity banner is the run-time proof.
  # `--color=no` matters: pytest colorizes `FAILED`/`ERROR` even when stdout is
  # a pipe, and the ANSI prefix stops `^FAILED` from ever matching. The first
  # version of this script omitted it and reported "0 failing" for a run with 4
  # real failures -- i.e. it failed GREEN, the same way #1106 itself did. The
  # ANSI strip below is belt-and-braces for anything else that colorizes.
  docker compose run --rm -v "$dir:/tree" train bash -lc \
    "cd /tree && python -m pytest ${PYTEST_ARGS[*]} -q --color=no --tb=line -rfE -p no:cacheprovider" \
    >"$WORK/$name.log" 2>&1 || pytest_status=$?
  docker compose run --rm -v "$dir:/tree" train bash -lc \
    "cd /tree && python -m pytest ${PYTEST_ARGS[*]} -q --color=no --collect-only -p no:cacheprovider" \
    >"$WORK/$name.collect.log" 2>&1 || collect_status=$?

  local strip_ansi=$'s/\033\\[[0-9;]*[a-zA-Z]//g'
  sed "$strip_ansi" "$WORK/$name.log" >"$WORK/$name.clean.log"
  sed "$strip_ansi" "$WORK/$name.collect.log" >"$WORK/$name.collect.clean.log"

  grep -E '^(FAILED|ERROR) ' "$WORK/$name.clean.log" | awk '{print $2}' | sort -u >"$WORK/$name.failed.txt" || true
  grep -E '^[^ ]+::' "$WORK/$name.collect.clean.log" | sort -u >"$WORK/$name.collected.txt" || true

  # Pytest exit 1 means real test failures and is expected input to the diff.
  # Every other non-zero status means the invocation itself broke (collection,
  # usage, interruption, or internal error) and must never become "0 failures".
  if (( pytest_status > 1 )); then
    echo "[compare_baseline] FATAL: $name pytest exited $pytest_status before a trustworthy comparison." >&2
    echo "  see $WORK/$name.log" >&2
    exit 4
  fi
  if (( collect_status != 0 )) || [[ ! -s "$WORK/$name.collected.txt" ]]; then
    echo "[compare_baseline] FATAL: $name collection failed or collected 0 tests." >&2
    echo "  see $WORK/$name.collect.log" >&2
    exit 4
  fi

  # Pytest's own summary line is the authoritative cross-check on the parsed
  # FAILED/ERROR detail lines. Merely printing it is insufficient: the first
  # implementation did that while still accepting an empty parsed set for a
  # real failing run. Any disagreement is a parser failure, so fail closed.
  local summary
  summary="$(grep -E '^[0-9]+ (failed|passed)|=+ .*(passed|failed)' "$WORK/$name.clean.log" | tail -1 || true)"
  if [[ -z "$summary" ]]; then
    echo "[compare_baseline] FATAL: $name has no recognizable pytest terminal summary." >&2
    echo "  see $WORK/$name.log" >&2
    exit 4
  fi
  local summary_failures parsed_failures
  summary_failures="$(grep -oE '[0-9]+ (failed|errors?)' <<<"$summary" | awk '{n += $1} END {print n + 0}')"
  parsed_failures="$(wc -l <"$WORK/$name.failed.txt")"
  if [[ "$summary_failures" -ne "$parsed_failures" ]]; then
    echo "[compare_baseline] FATAL: $name failure summary does not match parsed outcomes." >&2
    echo "  summary: $summary" >&2
    echo "  summary failures/errors: $summary_failures" >&2
    echo "  parsed FAILED/ERROR nodeids: $parsed_failures" >&2
    echo "  see $WORK/$name.log" >&2
    exit 4
  fi
  echo "[compare_baseline] $name summary: ${summary:-<none>}"

  # Property 3: surface the identity pytest reported, so a reader can see at a
  # glance that the two sides really ran different trees.
  local banner
  banner="$(grep -m1 'tree-identity:' "$WORK/$name.log" || true)"
  echo "[compare_baseline] $name banner: ${banner:-<none - conftest banner missing>}"
  echo "[compare_baseline] $name: $(wc -l <"$WORK/$name.failed.txt") failing, $(wc -l <"$WORK/$name.collected.txt") collected"
}

run_side baseline "$WORK/baseline" "$BASELINE_SHA"
run_side current "$WORK/current" "$CURRENT_SHA"

# --- Property 3 (hard check): identical banners means the same tree ran twice.
if [[ -s "$WORK/baseline.log" && -s "$WORK/current.log" ]]; then
  b_id="$(grep -m1 'tree-identity:' "$WORK/baseline.log" || true)"
  c_id="$(grep -m1 'tree-identity:' "$WORK/current.log" || true)"
  if [[ -n "$b_id" && "$b_id" == "$c_id" ]]; then
    echo "[compare_baseline] FATAL: both sides reported the same tree-identity banner:" >&2
    echo "  $b_id" >&2
    echo "  The comparison ran one tree twice. Results are meaningless." >&2
    exit 3
  fi
fi

# --- Partition ---------------------------------------------------------------
comm -12 "$WORK/baseline.collected.txt" "$WORK/current.collected.txt" >"$WORK/collected.in_both.txt"
comm -13 "$WORK/baseline.collected.txt" "$WORK/current.collected.txt" >"$WORK/collected.current_only.txt"
comm -23 "$WORK/baseline.collected.txt" "$WORK/current.collected.txt" >"$WORK/collected.baseline_only.txt"

# A genuine regression lives only in: collected on BOTH sides, passing at
# baseline, failing now. A test that is new on `current` failing is a
# different (still reportable) finding -- a merged PR shipping a red test.
comm -13 "$WORK/baseline.failed.txt" "$WORK/current.failed.txt" >"$WORK/failed.new.txt"
comm -23 "$WORK/baseline.failed.txt" "$WORK/current.failed.txt" >"$WORK/failed.fixed.txt"
comm -12 "$WORK/failed.new.txt" "$WORK/collected.in_both.txt" >"$WORK/regressions.txt"
comm -23 "$WORK/failed.new.txt" "$WORK/collected.in_both.txt" >"$WORK/failed.new_tests.txt"

echo
echo "================ compare_baseline summary ================"
echo "baseline $BASELINE_SHA : $(wc -l <"$WORK/baseline.failed.txt") failing / $(wc -l <"$WORK/baseline.collected.txt") collected"
echo "current  $CURRENT_SHA : $(wc -l <"$WORK/current.failed.txt") failing / $(wc -l <"$WORK/current.collected.txt") collected"
echo
echo "REGRESSIONS (collected on both sides, passed at baseline, fail now): $(wc -l <"$WORK/regressions.txt")"
sed 's/^/  ! /' "$WORK/regressions.txt"
echo
echo "NEW FAILING TESTS (did not exist at baseline): $(wc -l <"$WORK/failed.new_tests.txt")"
sed 's/^/  + /' "$WORK/failed.new_tests.txt"
echo
echo "NEWLY PASSING (failed at baseline, pass or gone now): $(wc -l <"$WORK/failed.fixed.txt")"
sed 's/^/  - /' "$WORK/failed.fixed.txt"
echo
echo "artifacts: $WORK"
echo "=========================================================="

if [[ -s "$WORK/regressions.txt" ]]; then
  exit 1
fi
exit 0
