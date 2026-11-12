"""Gating test for dark argparse options (issue #182).

A "dark option" is an argparse option declared in arguments/__init__.py but
never actually read by production code (and also not referenced in tests).

This test ensures:
1. No NEW dark options are added without explicit decision (via allowlist).
2. Existing dark options are tracked and documented.
3. Promotion/removal of dark options goes through the allowlist.

Reuses build_option_inventory.py's reader-scope logic to determine which
options have production_readers == 0.
"""
import sys
from pathlib import Path

# Torch-free: only import what build_option_inventory imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Known dark options (zero production readers) as of 2026-08-15.
# When an option is promoted, removed, or reclassified, update this allowlist
# AND comment the GitHub issue (#182) with the change.
#
# The first 13 were explicitly listed in issue #182's audit.
# Additional 4 were found by the build_option_inventory.py scan as of main HEAD.
DARK_OPTIONS_ALLOWLIST = {
    # Explicitly listed in #182 audit
    "geometry_prior_mode": "#182: zero production readers, test/doc-only mention",
    "hypersim_eval_hold": "#182: zero production readers, test/doc-only mention",
    "streaming_cover_voxel_multiplier": "#182: zero production readers, test/doc-only mention",
    "streaming_cover_voxel_size": "#182: zero production readers, test/doc-only mention",
    "streaming_covisible_min_shared": "#182: zero production readers, test/doc-only mention",
    "streaming_floater_consensus_min_baseline_m": "#182: zero production readers, test/doc-only mention",
    "streaming_insertion_debug_ply": "#182: zero production readers, test/doc-only mention",
    "streaming_keyframe_admit_eval_holdouts": "#182: zero production readers, test/doc-only mention",
    "streaming_keyframe_min_admit_frames": "#182: zero production readers, test/doc-only mention",
    "streaming_keyframe_rolling_window_size": "#182: zero production readers, test/doc-only mention",
    "streaming_support_window": "#182: zero production readers, test/doc-only mention",
    "surface_normal_max_depth_jump": "#182: zero production readers, test/doc-only mention",
    "surface_normal_min_valid_ratio": "#182: zero production readers, test/doc-only mention",
    "surface_normal_window": "#182: zero production readers, test/doc-only mention",
    # Additional dark options found by inventory scan
    "compute_cov3D_python": "zero production readers, legacy utility flag",
    "convert_SHs_python": "zero production readers, legacy utility flag",
    "streaming_floater_consensus_depth_tol_m": "zero production readers, unimplemented consensus feature",
    "streaming_keyframe_rolling_window_min_admits": "#313/#344: zero production readers, starvation floor inert since PR#20",
}


def test_no_new_dark_options():
    """Fail if a new dark option (zero production readers) appears not in allowlist.

    Imports build_option_inventory to reuse its production-reader-scope
    computation, avoiding duplication and the risk of divergence.

    A "dark option" is one with production_readers == 0, regardless of whether
    it appears in tests/docs.
    """
    from scripts.build_option_inventory import build

    # Build the full inventory using the canonical scanner.
    rows = build()

    # Identify all options with zero production readers (the "dark" category).
    dark_options = {
        row["name"] for row in rows if row["production_readers"] == 0
    }

    # Check for new dark options not in the allowlist.
    unknown_dark = dark_options - set(DARK_OPTIONS_ALLOWLIST.keys())
    if unknown_dark:
        msg = (
            f"Found {len(unknown_dark)} new option(s) with zero production readers not in the allowlist:\n"
            + "\n".join(f"  - {opt}" for opt in sorted(unknown_dark))
            + "\n\nAdd to DARK_OPTIONS_ALLOWLIST in this test, "
            + "then comment on issue #182 explaining the reclassification."
        )
        raise AssertionError(msg)

    # Also check that all allowlisted options still have zero production readers.
    # (If an option was promoted/removed, it should be removed from the allowlist.)
    gone_dark_options = set(DARK_OPTIONS_ALLOWLIST.keys()) - dark_options
    if gone_dark_options:
        msg = (
            f"The following options in the allowlist no longer have zero production readers:\n"
            + "\n".join(f"  - {opt}" for opt in sorted(gone_dark_options))
            + "\n\nRemove from DARK_OPTIONS_ALLOWLIST in this test, "
            + "then comment on issue #182 explaining the promotion/removal."
        )
        raise AssertionError(msg)


if __name__ == "__main__":
    test_no_new_dark_options()
    print("✓ All dark options accounted for in allowlist")
