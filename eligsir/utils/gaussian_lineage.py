"""Per-Gaussian lineage (birth method, parent/root UID, generation) helpers.

Separate from ``geometry_source`` (physical/geometric provenance, e.g.
sensor-derived vs. prior — already correctly inherited through MCMC descent).
``birth_method`` answers a different question: by which MECHANISM was this
specific Gaussian created (bootstrap / source-depth insertion / MCMC growth /
MCMC reseed). All functions here are pure and device-agnostic (device is
taken from the input tensors, never hardcoded), so they are unit-testable on
CPU without any GsplatGaussianModel/CUDA dependency.
"""

import torch

BIRTH_METHOD_BOOTSTRAP = 0
BIRTH_METHOD_SOURCE_DEPTH = 1
BIRTH_METHOD_MCMC_GROW = 2
BIRTH_METHOD_MCMC_RESEED = 3

_LINEAGE_BUFFER_NAMES = (
    "gaussian_uid", "parent_uid", "root_uid", "generation",
    "birth_method", "last_reseed_iter", "birth_iter",
)


def new_root_lineage_rows(gaussian_uid_slice: torch.Tensor, birth_method: int) -> dict:
    n = gaussian_uid_slice.shape[0]
    device = gaussian_uid_slice.device
    return {
        "parent_uid": torch.full((n,), -1, dtype=torch.int64, device=device),
        "root_uid": gaussian_uid_slice.clone(),
        "generation": torch.zeros(n, dtype=torch.int32, device=device),
        "birth_method": torch.full((n,), int(birth_method), dtype=torch.int8, device=device),
        "last_reseed_iter": torch.full((n,), -1, dtype=torch.int32, device=device),
    }


def growth_lineage_rows(
    existing_gaussian_uid: torch.Tensor,
    existing_root_uid: torch.Tensor,
    existing_generation: torch.Tensor,
    parent_idxs: torch.Tensor,
) -> dict:
    # existing_gaussian_uid must be each CURRENT row's own identity, not its
    # parent_uid — parent_idxs indexes into it to find the sampled parent's
    # own UID, which becomes the new clone's parent_uid.
    n = parent_idxs.shape[0]
    device = parent_idxs.device
    return {
        "parent_uid": existing_gaussian_uid[parent_idxs],
        "root_uid": existing_root_uid[parent_idxs],
        "generation": existing_generation[parent_idxs].to(torch.int32) + 1,
        "birth_method": torch.full((n,), BIRTH_METHOD_MCMC_GROW, dtype=torch.int8, device=device),
        "last_reseed_iter": torch.full((n,), -1, dtype=torch.int32, device=device),
    }


def reseed_lineage_update(
    gaussian_uid: torch.Tensor,
    root_uid: torch.Tensor,
    generation: torch.Tensor,
    dead_indices: torch.Tensor,
    sampled_idxs: torch.Tensor,
    iteration: int,
) -> dict:
    n = dead_indices.shape[0]
    device = dead_indices.device
    return {
        "parent_uid": gaussian_uid[sampled_idxs],
        "root_uid": root_uid[sampled_idxs],
        "generation": generation[sampled_idxs].to(torch.int32) + 1,
        "birth_method": torch.full((n,), BIRTH_METHOD_MCMC_RESEED, dtype=torch.int8, device=device),
        "last_reseed_iter": torch.full((n,), int(iteration), dtype=torch.int32, device=device),
    }


class SeedCohortUnavailable(RuntimeError):
    """The SEED (bootstrap) cohort cannot be determined EXACTLY for this model.

    Raised instead of returning a mask that would silently be wrong. See
    ``seed_cohort_mask`` for the failure cases (PLY-loaded model, pre-#283
    checkpoint, mid-run ``training_setup()`` re-bootstrap, legacy layout).
    """


def seed_uid_range(gaussians) -> tuple:
    """Return the ``[lo, hi)`` ``gaussian_uid`` block of the bootstrap population.

    Raises ``SeedCohortUnavailable`` when the model cannot testify to it.
    """
    rng = getattr(gaussians, "_seed_uid_range", None)
    if rng is None:
        reason = getattr(gaussians, "_seed_uid_range_reason", "") or (
            "this model does not track a SEED uid range at all (legacy "
            "GaussianModel layout has no per-Gaussian lineage buffers)"
        )
        raise SeedCohortUnavailable(
            f"SEED cohort is UNDETERMINED: {reason}. Refusing to return a "
            "contaminated cohort mask -- see issue #283."
        )
    return int(rng[0]), int(rng[1])


def _require_buffer(gaussians, name: str, n: int) -> torch.Tensor:
    tensor = getattr(gaussians, name, None)
    if tensor is None:
        raise SeedCohortUnavailable(
            f"SEED cohort is UNDETERMINED: required lineage buffer '{name}' is "
            "missing on this model."
        )
    if int(tensor.shape[0]) != n:
        raise SeedCohortUnavailable(
            f"SEED cohort is UNDETERMINED: lineage buffer '{name}' has "
            f"shape[0]={int(tensor.shape[0])} but the model has N={n} Gaussians "
            "-- a mutation site failed to keep it in lockstep."
        )
    return tensor


def seed_cohort_mask(gaussians) -> torch.Tensor:
    """Boolean mask over the CURRENT rows: surviving ORIGINAL bootstrap Gaussians.

    Exact by construction, and stable under every mutation the model performs:

    * ``gaussian_uid in [lo, hi)`` -- the bootstrap block is a contiguous
      ``arange`` allocated once from a monotonically increasing counter, so no
      later insertion or MCMC clone can ever land inside it. UIDs are only ever
      created (``arange``), mask-selected (``prune_points``) or concatenated
      (``append_running_state``) -- never remapped -- so this survives prunes,
      depth-insertion appends and growth appends unchanged.
    * ``root_uid == gaussian_uid and birth_method == BOOTSTRAP`` -- MCMC
      relocation (``reset_relocated_state``) deliberately KEEPS a revived slot's
      ``gaussian_uid`` ("capacity-slot identity never changes on reseed") while
      rewriting its ``root_uid``/``birth_method`` to the donor's. A uid-range
      test alone would therefore keep counting a relocated slot as a seed
      Gaussian even though the original one is gone; these two clauses drop it.

    Raises ``SeedCohortUnavailable`` rather than degrading silently.
    """
    lo, hi = seed_uid_range(gaussians)
    n = int(gaussians.get_xyz.shape[0])
    uid = _require_buffer(gaussians, "gaussian_uid", n)
    root = _require_buffer(gaussians, "root_uid", n)
    birth_method = _require_buffer(gaussians, "birth_method", n)
    return (
        (uid >= lo) & (uid < hi)
        & (root == uid)
        & (birth_method == BIRTH_METHOD_BOOTSTRAP)
    )


def seed_lineage_mask(gaussians) -> torch.Tensor:
    """Boolean mask: bootstrap Gaussians AND their MCMC descendants.

    ``root_uid`` is inherited by growth clones and by relocation donees, so
    ``root_uid in [lo, hi)`` is the closure of the seed cohort under descent.
    Superset of ``seed_cohort_mask``.
    """
    lo, hi = seed_uid_range(gaussians)
    n = int(gaussians.get_xyz.shape[0])
    root = _require_buffer(gaussians, "root_uid", n)
    return (root >= lo) & (root < hi)


def seed_cohort_summary(gaussians) -> dict:
    """Counts for #283 cohort telemetry. Raises if the cohort is undetermined."""
    lo, hi = seed_uid_range(gaussians)
    survivors = seed_cohort_mask(gaussians)
    lineage = seed_lineage_mask(gaussians)
    n = int(gaussians.get_xyz.shape[0])
    n_seed_born = hi - lo
    n_alive = int(survivors.sum().item())
    return {
        "seed_uid_lo": lo,
        "seed_uid_hi": hi,
        "n_seed_born": n_seed_born,
        "n_seed_alive": n_alive,
        "n_seed_lineage_alive": int(lineage.sum().item()),
        "seed_survival_fraction": (n_alive / n_seed_born) if n_seed_born else 0.0,
        "population": n,
    }


class CheckpointLineageView:
    """Duck-types the surface the cohort helpers above actually touch, over a
    ``chkpnt*.pth`` state dict (``torch.load(path)[0]``).

    Exists so that a STATIC checkpoint can be fed to the real
    ``seed_uid_range`` / ``seed_cohort_mask`` / ``seed_lineage_mask``
    primitives instead of any caller re-deriving the cohort by hand — the
    duplicate-definition problem issue #416 exists to retire. Both offline
    consumers (``scripts/seed_cohort_offline_replay.py`` and
    ``utils/seed_cohort_drift.py``) share this one view for that reason.

    ``get_opacity`` matches ``GsplatGaussianModel.get_opacity``
    (``sigmoid(params['opacities']).unsqueeze(-1)``) so a ``mean_opacity``
    computed here is the same number the TB logging site would have logged.

    ``_seed_uid_range_reason`` mirrors
    ``GsplatGaussianModel._restore_seed_uid_range`` (issue #416). Without it,
    ``seed_uid_range()`` falls back to its own default text — "this model does
    not track a SEED uid range at all (legacy GaussianModel layout has no
    per-Gaussian lineage buffers)" — which is FLATLY WRONG for a pre-#373
    gsplat checkpoint: it carries all seven lineage buffers and is simply
    missing the ``seed_uid_range`` key. Reporting a layout problem for a
    key-vintage problem sends the reader looking in the wrong place.
    """

    def __init__(self, blob: dict):
        self._seed_uid_range = blob.get("seed_uid_range")
        streaming_state = blob.get("streaming_state") or {}
        if not streaming_state:
            self._seed_uid_range_reason = (
                "checkpoint carries no streaming_state, so no lineage was restored"
            )
        elif self._seed_uid_range is None:
            self._seed_uid_range_reason = (
                "checkpoint predates the SEED uid range (no 'seed_uid_range' key); "
                "which uids were bootstrap rows is not recorded anywhere in it"
            )
        else:
            self._seed_uid_range_reason = ""
        for name in _LINEAGE_BUFFER_NAMES:
            setattr(self, name, streaming_state.get(name))
        self._means = blob["params"]["means"]
        self._opacities = blob["params"]["opacities"]

    @property
    def get_xyz(self):
        return self._means

    @property
    def get_opacity(self):
        return torch.sigmoid(self._opacities).unsqueeze(-1)


def warn_young_prot_mismatch_once(gaussians, site: str, warn_attr: str, birth_len: int, mask_len: int) -> None:
    """Print a one-time WARNING when young-Gaussian protection is skipped because
    ``birth_iter``'s length doesn't match the dead/kill mask's length, instead of
    silently no-op'ing the protection clause (see the birth-lineage correctness
    fix plan, Task 6, Finding 1 — this pattern is shared across three call sites:
    ``base._augment_dead_mask_knn_isolation``, ``energy_policy.compute_legacy_dead_mask``,
    and ``energy_mcmc.compute_dead_mask``).

    ``warn_attr`` is a distinct per-site attribute name stored on ``gaussians`` so
    the warning fires once per site (not once globally), and ``site`` is embedded
    in the printed message so which site fired is unambiguous from the log alone.
    """
    if getattr(gaussians, warn_attr, False):
        return
    print(
        f"[mcmc] WARNING: young-Gaussian protection skipped in {site} — "
        f"birth_iter.shape[0]={birth_len} != mask.shape[0]={mask_len}. This should not "
        f"happen after the birth-lineage correctness fix; if you see this, a mutation "
        f"site is not extending birth_iter correctly.",
        flush=True,
    )
    setattr(gaussians, warn_attr, True)


def assert_lineage_consistent(gaussians, context: str) -> None:
    n = int(gaussians.get_xyz.shape[0])
    mismatched = []
    for name in _LINEAGE_BUFFER_NAMES:
        tensor = getattr(gaussians, name, None)
        if tensor is None:
            continue
        if tensor.shape[0] != n:
            mismatched.append(f"{name} (shape[0]={tensor.shape[0]})")
    if mismatched:
        raise RuntimeError(
            f"[gaussian_lineage] topology inconsistency at '{context}': expected N={n}, "
            f"mismatched buffers: {', '.join(mismatched)}"
        )
