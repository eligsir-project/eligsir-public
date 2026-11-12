"""
Compiled kernel wrappers for tensor-only helpers.

Provides eager + torch.compile variants with graceful ROCm fallback.
Compile only after ``compile_after_iter`` to avoid growth-phase churn.

Defaults to **off**. Enable per-kernel via CLI flags (e.g. ``--compile_sh``).

Machine-readable status (issue #597)
------------------------------------
``compile_mode=off`` used to leave nothing but a console line, so a finished
run could not be asked whether compilation was *intentionally* disabled, was
unsupported, or was requested and silently fell back to eager -- three states
with the same observable throughput story and very different meanings for a
performance claim. :func:`build_compute_backend_block` returns the resolved
record (written into ``report.json["compute_backend"]`` by
``utils/comparison_report.py``), covering both Torch graph compilation and the
fused/custom kernels that stay active regardless of it.
"""
import copy
import torch

# ---------------------------------------------------------------------------
# Compilation availability & ROCm detection
# ---------------------------------------------------------------------------
_COMPILE_AVAILABLE = hasattr(torch, "compile")
_IS_ROCM = False
try:
    _IS_ROCM = torch.version.hip is not None
except Exception:
    pass


# Set by _compile_or_eager on a failed compile and consumed immediately by the
# caller below, which knows the REGISTRY name the function was registered under
# (``fn.__name__`` is the private eager symbol, not that key).
_LAST_COMPILE_ERROR = {"reason": ""}


def _compile_or_eager(fn, enabled=True, mode="reduce-overhead", dynamic=True):
    """
    Try ``torch.compile(fn)``; on any failure return the eager function.
    """
    _LAST_COMPILE_ERROR["reason"] = ""
    if not _COMPILE_AVAILABLE or not enabled:
        return fn
    try:
        compiled = torch.compile(fn, mode=mode, dynamic=dynamic, fullgraph=False)
        return compiled
    except Exception as exc:
        # Loud, and RECORDED (issue #597): a requested compile that falls back
        # must never end up indistinguishable from an intentional
        # --compile_mode=off.
        print(
            f"[compiled-kernels] WARNING: torch.compile FAILED for {fn.__name__}: "
            f"{exc!r}\n"
            f"[compiled-kernels] Falling back to eager. This run is NOT an "
            f"intentionally-uncompiled run -- see report.json['compute_backend'].",
            flush=True,
        )
        _LAST_COMPILE_ERROR["reason"] = repr(exc)
        return fn


# ---------------------------------------------------------------------------
# Registry that switches eager -> compiled after a configurable iteration.
# ---------------------------------------------------------------------------
class _CompiledKernelRegistry:
    def __init__(self):
        self._eager = {}
        self._compiled = {}
        self._active = {}
        self._enabled_globally = True
        self._mode = "reduce-overhead"
        self._dynamic = True
        self._compile_after_iter = 0
        self._current_iter = 0
        self._compiled_once = set()
        # issue #597: per-kernel outcome, keyed by REGISTRY name.
        # {"requested": bool, "compiled": bool|None, "compile_error": str,
        #  "runtime_error": str}
        self._outcomes = {}

    def configure(self, enabled=True, mode="reduce-overhead", dynamic=True, compile_after_iter=0):
        self._enabled_globally = enabled
        self._mode = mode
        self._dynamic = dynamic
        self._compile_after_iter = compile_after_iter

    def set_iteration(self, iteration):
        self._current_iter = iteration

    def register(self, name, eager_fn, enabled=True):
        self._eager[name] = eager_fn
        self._active[name] = enabled
        # compiled version is created lazily on first use after compile_after_iter
        self._compiled.pop(name, None)
        self._compiled_once.discard(name)
        self._outcomes[name] = {
            "requested": bool(enabled),
            # None = never attempted yet (lazy: only on first call past
            # compile_after_iter). Distinct from False = attempted and failed.
            "compiled": None,
            "compile_error": "",
            "runtime_error": "",
        }
        return self

    def kernel_outcomes(self):
        """Snapshot of every registered kernel's compile outcome (issue #597)."""
        return copy.deepcopy(self._outcomes)

    def _get_compiled(self, name):
        if name not in self._compiled:
            eager_fn = self._eager[name]
            compiled = _compile_or_eager(
                eager_fn,
                enabled=self._active.get(name, True) and self._enabled_globally,
                mode=self._mode,
                dynamic=self._dynamic,
            )
            self._compiled[name] = compiled
            outcome = self._outcomes.setdefault(
                name,
                {"requested": bool(self._active.get(name, True)), "compiled": None,
                 "compile_error": "", "runtime_error": ""},
            )
            if compiled is not eager_fn:
                outcome["compiled"] = True
            elif self._active.get(name, True) and self._enabled_globally:
                # It was requested, an attempt was made, and we still have the
                # eager function back -- a real fallback, not a no-op.
                outcome["compiled"] = False
                outcome["compile_error"] = (
                    _LAST_COMPILE_ERROR["reason"]
                    or ("torch.compile unavailable" if not _COMPILE_AVAILABLE else "")
                )
            if compiled is not eager_fn and name not in self._compiled_once:
                self._compiled_once.add(name)
                print(
                    f"[compiled-kernels] Compiled '{name}' "
                    f"(mode={self._mode}, dynamic={self._dynamic})"
                )
        return self._compiled[name]

    def __call__(self, name, *args, **kwargs):
        if self._current_iter < self._compile_after_iter:
            return self._eager[name](*args, **kwargs)
        try:
            return self._get_compiled(name)(*args, **kwargs)
        except Exception as exc:
            print(
                f"[compiled-kernels] WARNING: runtime failure in '{name}': {exc!r}\n"
                f"[compiled-kernels] Disabling '{name}', falling back to eager. "
                f"Recorded in report.json['compute_backend'].",
                flush=True,
            )
            self._compiled[name] = self._eager[name]
            self._active[name] = False
            outcome = self._outcomes.setdefault(
                name,
                {"requested": True, "compiled": None, "compile_error": "",
                 "runtime_error": ""},
            )
            outcome["compiled"] = False
            outcome["runtime_error"] = repr(exc)
            return self._eager[name](*args, **kwargs)

    def eager(self, name, *args, **kwargs):
        """Force eager execution (useful for debugging / graph breaks)."""
        return self._eager[name](*args, **kwargs)


REGISTRY = _CompiledKernelRegistry()


# ---------------------------------------------------------------------------
# Eager kernel implementations
# ---------------------------------------------------------------------------

def _apply_sh_clamp(colors, clamp_mode="clamped"):
    if clamp_mode == "clamped":
        return torch.clamp(colors + 0.5, 0.0, 1.0).contiguous()
    if clamp_mode == "lower_only":
        return torch.clamp_min(colors + 0.5, 0.0).contiguous()
    raise ValueError(
        f"Unsupported sh_color_clamp_mode '{clamp_mode}'. Expected 'clamped' or 'lower_only'."
    )


def _sh_to_rgb_deg1(features_dc, features_rest, dir_pp_normalized, clamp_mode="clamped"):
    """
    Full SH-to-RGB for degree 1.
    Takes full [N, 15, 3] features_rest, slices internally to degree-1 coeffs.
    """
    from eligsir.utils.sh_utils import eval_sh
    dc = features_dc.transpose(1, 2)                         # [N, 3, 1]
    rest = features_rest[:, :3, :].transpose(1, 2)           # [N, 3, 3]
    shs = torch.cat((dc, rest), dim=2).contiguous()          # [N, 3, 4]
    colors = eval_sh(1, shs, dir_pp_normalized)
    return _apply_sh_clamp(colors, clamp_mode)


def _sh_to_rgb_deg2(features_dc, features_rest, dir_pp_normalized, clamp_mode="clamped"):
    """
    Full SH-to-RGB for degree 2.
    Takes full [N, 15, 3] features_rest, slices internally to degree-2 coeffs.
    """
    from eligsir.utils.sh_utils import eval_sh
    dc = features_dc.transpose(1, 2)
    rest = features_rest[:, :8, :].transpose(1, 2)
    shs = torch.cat((dc, rest), dim=2).contiguous()
    colors = eval_sh(2, shs, dir_pp_normalized)
    return _apply_sh_clamp(colors, clamp_mode)


def _sh_to_rgb_deg3(features_dc, features_rest, dir_pp_normalized, clamp_mode="clamped"):
    """
    Full SH-to-RGB for degree 3.
    Takes full [N, 15, 3] features_rest, uses all 15 rest coeffs.
    """
    from eligsir.utils.sh_utils import eval_sh
    dc = features_dc.transpose(1, 2)
    rest = features_rest[:, :15, :].transpose(1, 2)
    shs = torch.cat((dc, rest), dim=2).contiguous()
    colors = eval_sh(3, shs, dir_pp_normalized)
    return _apply_sh_clamp(colors, clamp_mode)


def _effective_count_core(opacities, dead_threshold, softness):
    """
    Core of compute_effective_count.
    Args:
        opacities: [N, 1] raw opacity params
    Returns:
        scalar N_eff
    """
    alpha = torch.sigmoid(opacities).squeeze(-1)
    return torch.sigmoid((alpha - dead_threshold) / softness).sum()


def _effective_count_loss_core(opacities, target, target_scale, dead_threshold, softness):
    """
    Core of compute_effective_count_loss.
    Returns (loss, n_eff) tuple — avoids recomputing N_eff for logging.
    """
    alpha = torch.sigmoid(opacities).squeeze(-1)
    n_eff = torch.sigmoid((alpha - dead_threshold) / softness).sum()
    loss = ((n_eff - target) / target_scale) ** 2
    return loss, n_eff


def _opacity_entropy_core(opacities):
    """
    Core of compute_opacity_entropy_loss.
    Args:
        opacities: [N, 1] raw
    Returns:
        scalar mean entropy
    """
    alpha = torch.sigmoid(opacities).squeeze(-1)
    eps = 1e-6
    entropy = -alpha * torch.log(alpha + eps) - (1.0 - alpha) * torch.log(1.0 - alpha + eps)
    return entropy.mean()


def _utility_core(
    alpha, v, support, xyz_grad, opacity_grad, scale_grad,
    max_scale, w_alpha, w_vis, w_support, w_grad, w_scale,
    w_dead, beta_opacity, beta_scale, alpha_dead,
):
    """
    Pure-tensor core of compute_gaussian_utility.
    All inputs are 1-D tensors of length N (or scalars for weights).
    Returns:
        [N] utility scores
    """
    norm_grad = xyz_grad + beta_opacity * opacity_grad + beta_scale * scale_grad
    scale_penalty = torch.clamp(max_scale - 0.1, min=0.0) ** 2
    dead_penalty = (alpha < alpha_dead).float()
    return (
        w_alpha * alpha + w_vis * v + w_support * support + w_grad * norm_grad
        - w_scale * scale_penalty - w_dead * dead_penalty
    )


def _active_reg_core(opacity, scaling, w_opacity, w_scale, w_aniso=0.0, aniso_max=10.0):
    """
    Active-set L1 regularizer core.
    NOTE: Always runs eager — active-set shapes change every iteration.

    w_aniso > 0 adds a penalty on the max/min scale ratio, discouraging
    needle-shaped Gaussians that cause axis-aligned MCMC drift.
    """
    if opacity.numel() == 0 or scaling.numel() == 0:
        return scaling.sum() * 0.0
    loss = w_opacity * torch.abs(opacity).mean() + w_scale * torch.abs(scaling).mean()
    if w_aniso > 0.0 and scaling.shape[-1] >= 2:
        s_max = scaling.max(dim=-1).values
        s_min = scaling.min(dim=-1).values.clamp(min=1e-6)
        ratio = (s_max / s_min).clamp(max=aniso_max)
        loss = loss + w_aniso * ratio.mean()
    return loss


def _active_reg_core_masked(opacity, scaling, mask, w_opacity, w_scale, w_aniso=0.0, aniso_max=10.0):
    """
    Fixed-shape, full-population equivalent of :func:`_active_reg_core` for a
    boolean active-set ``mask`` (issue #1670).

    The caller of ``_active_reg_core`` used to slice its own active subset
    first (``opacity[mask], scaling[mask]``) — a data-dependent boolean
    compaction whose output cardinality changes every iteration and forces a
    new CUDA allocation each time. This computes the exact same weighted-mean
    quantities over the FULL ``[N, ...]`` population instead, using ``mask``
    only as a multiplicative device weight, so shape is fixed across calls.

    Parity (see ``tests/test_active_reg_core_masked_parity.py``):

    - non-empty mask: reproduces ``_active_reg_core(opacity[mask],
      scaling[mask], ...)`` up to floating-point reduction-order tolerance —
      the masked-out rows contribute an exact ``0.0`` addend to every sum, so
      the only difference from the compacted version is summation order;
    - empty mask (``mask.sum() == 0``): returns an exact connected device
      zero (the denominator is clamped to 1, never 0/0), matching
      ``_active_reg_core``'s own empty-subset zero;
    - gradients outside the mask are exactly zero: each masked-out row's
      local gradient (finite, since Gaussian parameters are finite/sanitized
      state, not an unproven external measurement) is multiplied by that
      row's constant ``0.0`` mask weight, which is an ordinary zero-times-
      finite product — not a NaN-producing term being papered over after the
      fact (contrast the depth/total-loss finite gates elsewhere, which
      guard against genuinely non-finite aggregate losses and are NOT
      touched by this helper).

    Perf note (2026-09-03 throughput-recovery follow-up): avoids a redundant
    mask-to-dtype cast when opacity and scaling share a dtype (the common
    case), instead of computing it twice. This is a small reduction in
    per-call kernel-launch count, not a reduction in abs/multiply/sum op
    count -- the arithmetic itself is unchanged. Exact-parity behavior with
    the prior version is unchanged and still pinned by
    tests/test_active_reg_core_masked_parity.py.
    """
    count = mask.sum().clamp(min=1)
    mask_opacity = mask.to(dtype=opacity.dtype).unsqueeze(-1)
    opacity_term = (opacity.abs() * mask_opacity).sum() / (count * opacity.shape[-1])
    mask_scaling = (
        mask_opacity
        if scaling.dtype == opacity.dtype
        else mask.to(dtype=scaling.dtype).unsqueeze(-1)
    )
    scale_term = (scaling.abs() * mask_scaling).sum() / (count * scaling.shape[-1])
    loss = w_opacity * opacity_term + w_scale * scale_term
    if w_aniso > 0.0 and scaling.shape[-1] >= 2:
        s_max = scaling.max(dim=-1).values
        s_min = scaling.min(dim=-1).values.clamp(min=1e-6)
        ratio = (s_max / s_min).clamp(max=aniso_max)
        mask_1d = mask_scaling.squeeze(-1) if scaling.dtype == opacity.dtype else mask.to(dtype=scaling.dtype)
        loss = loss + w_aniso * (ratio * mask_1d).sum() / count
    return loss


# ---------------------------------------------------------------------------
# Public API — functions that dispatch through the registry
# ---------------------------------------------------------------------------

def sh_to_rgb(deg, features_dc, features_rest, dir_pp_normalized, clamp_mode="clamped"):
    """
    Full SH-to-RGB pipeline (transpose → slice → cat → eval_sh → clamp → contiguous).

    Features rest is always passed as full [N, 15, 3] regardless of active degree;
    the degree-specific wrapper slices internally. This keeps input shapes stable
    across SH degree changes, avoiding recompilation.

    Args:
        deg: SH degree (1-3). Degree 0 callers should use the inline path.
        features_dc: [N, 1, 3] raw DC features
        features_rest: [N, 15, 3] full rest features (caller may have detached)
        dir_pp_normalized: [N, 3] unit view directions
        clamp_mode: "clamped" (default) or "lower_only"
    Returns:
        [N, 3] clamped RGB colors
    """
    if clamp_mode != "clamped":
        if deg == 1:
            return _sh_to_rgb_deg1(features_dc, features_rest, dir_pp_normalized, clamp_mode=clamp_mode)
        elif deg == 2:
            return _sh_to_rgb_deg2(features_dc, features_rest, dir_pp_normalized, clamp_mode=clamp_mode)
        elif deg == 3:
            return _sh_to_rgb_deg3(features_dc, features_rest, dir_pp_normalized, clamp_mode=clamp_mode)
        else:
            raise ValueError(f"Unsupported SH degree for compiled path: {deg}")
    if deg == 1:
        return REGISTRY("sh_to_rgb_deg1", features_dc, features_rest, dir_pp_normalized)
    elif deg == 2:
        return REGISTRY("sh_to_rgb_deg2", features_dc, features_rest, dir_pp_normalized)
    elif deg == 3:
        return REGISTRY("sh_to_rgb_deg3", features_dc, features_rest, dir_pp_normalized)
    else:
        raise ValueError(f"Unsupported SH degree for compiled path: {deg}")


def sh_to_rgb_eager(deg, features_dc, features_rest, dir_pp_normalized, clamp_mode="clamped"):
    if clamp_mode == "clamped":
        if deg == 1:
            if "sh_to_rgb_deg1" in REGISTRY._eager:
                return REGISTRY.eager("sh_to_rgb_deg1", features_dc, features_rest, dir_pp_normalized)
            return _sh_to_rgb_deg1(features_dc, features_rest, dir_pp_normalized)
        elif deg == 2:
            if "sh_to_rgb_deg2" in REGISTRY._eager:
                return REGISTRY.eager("sh_to_rgb_deg2", features_dc, features_rest, dir_pp_normalized)
            return _sh_to_rgb_deg2(features_dc, features_rest, dir_pp_normalized)
        elif deg == 3:
            if "sh_to_rgb_deg3" in REGISTRY._eager:
                return REGISTRY.eager("sh_to_rgb_deg3", features_dc, features_rest, dir_pp_normalized)
            return _sh_to_rgb_deg3(features_dc, features_rest, dir_pp_normalized)
        else:
            raise ValueError(f"Unsupported SH degree for eager path: {deg}")
    if deg == 1:
        return _sh_to_rgb_deg1(features_dc, features_rest, dir_pp_normalized, clamp_mode=clamp_mode)
    if deg == 2:
        return _sh_to_rgb_deg2(features_dc, features_rest, dir_pp_normalized, clamp_mode=clamp_mode)
    if deg == 3:
        return _sh_to_rgb_deg3(features_dc, features_rest, dir_pp_normalized, clamp_mode=clamp_mode)
    raise ValueError(f"Unsupported SH degree for eager path: {deg}")


def effective_count_core(opacities, dead_threshold=0.005, softness=0.002):
    return REGISTRY("effective_count_core", opacities, dead_threshold, softness)


def effective_count_loss_core(opacities, target, target_scale, dead_threshold=0.005, softness=0.002):
    """Returns (loss, n_eff) tuple."""
    return REGISTRY(
        "effective_count_loss_core", opacities, target, target_scale,
        dead_threshold, softness,
    )


def opacity_entropy_core(opacities):
    return REGISTRY("opacity_entropy_core", opacities)


def utility_core(
    alpha, v, support, xyz_grad, opacity_grad, scale_grad,
    max_scale, w_alpha=1.0, w_vis=2.0, w_support=2.0, w_grad=3.0,
    w_scale=0.5, w_dead=1.0, beta_opacity=1.0, beta_scale=0.5, alpha_dead=0.005,
):
    return REGISTRY(
        "utility_core", alpha, v, support, xyz_grad, opacity_grad, scale_grad,
        max_scale, w_alpha, w_vis, w_support, w_grad, w_scale, w_dead,
        beta_opacity, beta_scale, alpha_dead,
    )


def active_reg_core(opacity, scaling, w_opacity=0.01, w_scale=0.01, w_aniso=0.0, aniso_max=10.0):
    """Active-set L1 regularizer — always eager (shape changes every iteration)."""
    return _active_reg_core(opacity, scaling, w_opacity, w_scale, w_aniso=w_aniso, aniso_max=aniso_max)


def active_reg_core_masked(opacity, scaling, mask, w_opacity=0.01, w_scale=0.01, w_aniso=0.0, aniso_max=10.0):
    """Fixed-shape active-set L1 regularizer: full-population ``opacity``/
    ``scaling`` plus a boolean ``mask``, instead of the caller pre-slicing
    ``opacity[mask]``/``scaling[mask]`` (issue #1670). See
    ``_active_reg_core_masked`` for the exact parity/gradient argument."""
    return _active_reg_core_masked(opacity, scaling, mask, w_opacity, w_scale, w_aniso=w_aniso, aniso_max=aniso_max)


def young_isotropy_ramp(age, ramp_frames):
    """Pure ramp weight for the decaying isotropy regularizer on immature
    (provisional+young) Gaussians (CPU/CUDA agnostic).

    ramp(age) = clamp(1 - age/ramp_frames, 0, 1): 1.0 at birth (age<=0),
    linearly decaying to 0.0 at age>=ramp_frames. Accepts either a python
    scalar or a torch tensor for ``age``; ``ramp_frames`` is a scalar.
    """
    if ramp_frames <= 0:
        # Degenerate ramp window: fully decayed immediately.
        if isinstance(age, torch.Tensor):
            return torch.zeros_like(age, dtype=torch.float32)
        return 0.0
    if isinstance(age, torch.Tensor):
        return (1.0 - age.float() / float(ramp_frames)).clamp(0.0, 1.0)
    return max(0.0, min(1.0, 1.0 - float(age) / float(ramp_frames)))


def young_opacity_reg_weights(
    age,
    support_count,
    young_exempt_iters: int,
    maturity_ramp: bool,
    target_support: float,
    ramp_min: float,
    recency_gap=None,
    recency_horizon: float = 0.0,
    recency_ramp_min: float = 0.2,
):
    """Per-Gaussian opacity-reg weight in [0, 1] (pure; CPU/CUDA agnostic).

    Composes three default-off relaxations of opacity regularization for the
    live streaming path:

    * Young exemption: Gaussians with ``age < young_exempt_iters`` get weight 0
      (fully exempt). ``age`` is ``iteration - birth_iter`` per Gaussian.
      Disabled when ``young_exempt_iters <= 0``.
    * Maturity ramp: weight = ``clamp(support_count / target_support, ramp_min,
      1.0)`` so newborns get ``ramp_min`` pressure and well-supervised Gaussians
      get full pressure. Disabled when ``maturity_ramp`` is False.
    * Recency ramp: weight = ``clamp(1 - recency_gap / recency_horizon,
      recency_ramp_min, 1.0)``, where ``recency_gap`` is
      ``iteration - last_supervised_iter`` per Gaussian -- so a Gaussian that
      hasn't been in a training batch (visible in the sampled camera's render)
      for a while gets reduced opacity-reg pressure, not just newborns.
      Disabled when ``recency_horizon <= 0``.

    When all are off the returned weights are all 1.0 (full opacity reg —
    identical to ``active_reg_core(w_opacity=opacity_reg)``). All compose
    multiplicatively: an exempt young Gaussian stays at 0 regardless of the
    other two factors.
    """
    if maturity_ramp and support_count is not None:
        sc = support_count.float()
        w = (sc / max(float(target_support), 1e-6)).clamp(float(ramp_min), 1.0)
    else:
        w = torch.ones_like(age, dtype=torch.float32)
    if young_exempt_iters > 0 and age is not None:
        w = w * (age >= int(young_exempt_iters)).to(w.dtype)
    if recency_horizon > 0 and recency_gap is not None:
        recency_w = (1.0 - recency_gap.float() / float(recency_horizon)).clamp(
            float(recency_ramp_min), 1.0
        )
        w = w * recency_w
    return w


# ---------------------------------------------------------------------------
# Setup / configuration helpers
# ---------------------------------------------------------------------------

# --- issue #597: machine-readable compile/backend status --------------------
#
# Resolved compile-status vocabulary. `unsupported` is defined because #597
# asks for it, but NO CURRENT CODE PATH PRODUCES IT: `_COMPILE_AVAILABLE` is
# `hasattr(torch, "compile")`, true on every torch this repo runs, and ROCm is
# warned about rather than refused. It is here so a future
# refuse-to-compile path has a name that is not `disabled_by_config`.
COMPILE_STATUS_NOT_CONFIGURED = "not_configured"
COMPILE_STATUS_DISABLED_BY_CONFIG = "disabled_by_config"
COMPILE_STATUS_UNSUPPORTED = "unsupported"
COMPILE_STATUS_ENABLED = "enabled"
COMPILE_STATUS_PARTIALLY_COMPILED = "partially_compiled"
COMPILE_STATUS_COMPILE_FAILED_FALLBACK = "compile_failed_fallback"
COMPILE_STATUS_REQUESTED_NO_KERNELS = "requested_no_kernels_selected"
COMPILE_STATUS_REQUESTED_NOT_YET_COMPILED = "requested_not_yet_compiled"

# Populated by configure_torch_compile(); read by build_compute_backend_block().
# `None` means configure_torch_compile() never ran in this process, which is a
# real, reportable state (a bare report/render tool), not "compilation off".
_COMPILE_CONFIG = None


_MISSING = object()


def _observed(args, name):
    """Read ``args.name`` for the RECORD, never substituting a default.

    Deliberately not ``getattr(args, name, <literal>)``: issue #368's
    repo-wide guard (``tests/test_getattr_fallback_freshness.py``) exists
    because such literals drift from ``arguments/__init__.py``'s real
    defaults, and a *reporting* block that fabricates ``optimizer_type="adam"``
    for a run that really used ``selective_adam`` is worse than one that says
    it does not know. Absent attribute -> ``None`` -> "not observed".
    """
    value = getattr(args, name, _MISSING)
    return None if value is _MISSING else value


def _requested_ssim_backend(args):
    """The SSIM backend this args namespace asks for.

    Mirrors the identical if/elif in train.py and train_streaming.py, except
    that a namespace carrying NEITHER flag reports ``None`` rather than
    claiming ``ref``. The RESOLVED backend is read separately from
    ``utils.loss_utils._SSIM_MODE`` (``set_ssim_mode`` falls back to ``ref``
    when fused_ssim is missing), so requested-vs-resolved is visible the same
    way it is for compile_mode.
    """
    fused = _observed(args, "fused_ssim")
    compiled = _observed(args, "compile_loss")
    if fused is None and compiled is None:
        return None
    if bool(fused):
        return "fused"
    if bool(compiled):
        return "compiled"
    return "ref"


def _resolved_ssim_backend():
    try:
        from eligsir.utils import loss_utils

        return str(getattr(loss_utils, "_SSIM_MODE", "") or "")
    except Exception:  # pragma: no cover - defensive
        return ""


def resolve_compile_status(config=None, outcomes=None):
    """Resolve the single compile-status label for this run (issue #597)."""

    config = _COMPILE_CONFIG if config is None else config
    if config is None:
        return COMPILE_STATUS_NOT_CONFIGURED
    if str(config.get("requested_compile_mode", "off")) == "off":
        return COMPILE_STATUS_DISABLED_BY_CONFIG
    if not config.get("torch_compile_available", True):
        return COMPILE_STATUS_UNSUPPORTED

    outcomes = REGISTRY.kernel_outcomes() if outcomes is None else outcomes
    requested = {k: v for k, v in outcomes.items() if v.get("requested")}
    if not requested:
        # compile_mode was set but every per-kernel flag (--compile_sh,
        # --compile_energy, --compile_utility) stayed off, so nothing was ever
        # eligible. Emphatically NOT the same as compile_mode=off.
        return COMPILE_STATUS_REQUESTED_NO_KERNELS
    attempted = {k: v for k, v in requested.items() if v.get("compiled") is not None}
    if not attempted:
        # compile_after_iter was never reached (the growth-aware default puts
        # it past mcmc_stop_growth_iter, so a short run legitimately ends here).
        return COMPILE_STATUS_REQUESTED_NOT_YET_COMPILED
    ok = [k for k, v in attempted.items() if v.get("compiled")]
    if not ok:
        return COMPILE_STATUS_COMPILE_FAILED_FALLBACK
    if len(ok) == len(requested):
        return COMPILE_STATUS_ENABLED
    return COMPILE_STATUS_PARTIALLY_COMPILED


def build_compute_backend_block():
    """The ``report.json["compute_backend"]`` record (issue #597).

    A pure accessor over process state -- it configures nothing and never
    raises. Safe to call from the report writer of a process that never
    trained (it then reports ``not_configured``).
    """

    config = copy.deepcopy(_COMPILE_CONFIG) if _COMPILE_CONFIG else None
    outcomes = REGISTRY.kernel_outcomes()
    status = resolve_compile_status(config, outcomes)
    block = {
        "schema_version": 1,
        "configured": config is not None,
        "compile_status": status,
        # Requested vs resolved, deliberately separate fields.
        "requested_compile_mode": (
            config.get("requested_compile_mode") if config else None
        ),
        "resolved_compile_mode": (
            config.get("resolved_compile_mode") if config else None
        ),
        "compile_dynamic_requested": (
            config.get("compile_dynamic_requested") if config else None
        ),
        "compile_dynamic_resolved": (
            config.get("compile_dynamic_resolved") if config else None
        ),
        "compile_after_iter": config.get("compile_after_iter") if config else None,
        "torch_compile_available": (
            config.get("torch_compile_available") if config else _COMPILE_AVAILABLE
        ),
        "is_rocm": config.get("is_rocm") if config else _IS_ROCM,
        "kernels": outcomes,
        "compile_failures": {
            name: {
                "compile_error": data.get("compile_error", ""),
                "runtime_error": data.get("runtime_error", ""),
            }
            for name, data in outcomes.items()
            if data.get("compile_error") or data.get("runtime_error")
        },
        # Fused/custom kernels are INDEPENDENT of torch.compile: a run with
        # compile_status=disabled_by_config can still be running fused SSIM
        # and selective_adam. #597 acceptance criterion 2.
        "fused_kernels": {
            "ssim_backend_requested": config.get("ssim_backend_requested") if config else None,
            "ssim_backend_resolved": _resolved_ssim_backend() or None,
            "optimizer_type": config.get("optimizer_type") if config else None,
            "gsplat_sparse_grad": config.get("gsplat_sparse_grad") if config else None,
            "sh_update_interval": config.get("sh_update_interval") if config else None,
            "parallelism_profile": config.get("parallelism_profile") if config else None,
            "sh_impl": config.get("sh_impl") if config else None,
        },
    }
    block["torch_compilation_active"] = status in (
        COMPILE_STATUS_ENABLED,
        COMPILE_STATUS_PARTIALLY_COMPILED,
    )
    block["compile_requested_but_not_active"] = status in (
        COMPILE_STATUS_UNSUPPORTED,
        COMPILE_STATUS_COMPILE_FAILED_FALLBACK,
        COMPILE_STATUS_REQUESTED_NO_KERNELS,
        COMPILE_STATUS_REQUESTED_NOT_YET_COMPILED,
    )
    return block


def compute_backend_differences(current, baseline):
    """Field-level diff of two ``compute_backend`` blocks (issue #597 AC 3).

    Plain-data comparison so performance tooling can surface a compile/backend
    mismatch between two runs. Either side may be ``None`` (a report written
    before this block existed), which is itself reported.
    """

    if not isinstance(current, dict) or not isinstance(baseline, dict):
        return {
            "comparable": False,
            "reason": (
                "one or both reports carry no compute_backend block "
                "(written before issue #597)"
            ),
            "differences": {},
        }
    diffs = {}
    for key in ("compile_status", "resolved_compile_mode", "compile_dynamic_resolved",
                "compile_after_iter", "is_rocm", "torch_compilation_active"):
        if current.get(key) != baseline.get(key):
            diffs[key] = {"current": current.get(key), "baseline": baseline.get(key)}
    cur_fused = current.get("fused_kernels") or {}
    base_fused = baseline.get("fused_kernels") or {}
    for key in sorted(set(cur_fused) | set(base_fused)):
        if cur_fused.get(key) != base_fused.get(key):
            diffs[f"fused_kernels.{key}"] = {
                "current": cur_fused.get(key), "baseline": base_fused.get(key),
            }
    return {
        "comparable": True,
        "reason": "",
        "differences": diffs,
        "throughput_comparable": not diffs,
    }


def format_backend_summary(block):
    """Startup summary separating Torch graph compilation from fused kernels."""

    fused = block.get("fused_kernels") or {}
    lines = [
        "[backend] ---- compute backend ----",
        f"[backend] torch.compile: status={block.get('compile_status')} "
        f"requested_mode={block.get('requested_compile_mode')} "
        f"resolved_mode={block.get('resolved_compile_mode')} "
        f"dynamic={block.get('compile_dynamic_resolved')} "
        f"after_iter={block.get('compile_after_iter')} "
        f"rocm={block.get('is_rocm')}",
        "[backend] fused/custom kernels (INDEPENDENT of torch.compile): "
        f"ssim={fused.get('ssim_backend_requested')} "
        f"optimizer={fused.get('optimizer_type')} "
        f"sparse_grad={fused.get('gsplat_sparse_grad')} "
        f"sh_update_interval={fused.get('sh_update_interval')} "
        f"profile={fused.get('parallelism_profile')}",
    ]
    if block.get("compile_status") == COMPILE_STATUS_DISABLED_BY_CONFIG:
        lines.append(
            "[backend] NOTE: torch.compile is off by explicit request; the "
            "fused/custom kernels above are UNAFFECTED and may still be active."
        )
    return "\n".join(lines)


def configure_torch_compile(args):
    """
    Call once at training start with the parsed arg namespace.

    Reads ``compile_mode``, ``compile_dynamic``, ``compile_after_iter``,
    and per-kernel flags.

    Kernels are **always** registered so that eager fallback works even
    when ``compile_mode=off``.
    """
    # --- Register all kernels (eager fallback always needed) ---
    compile_sh = bool(getattr(args, "compile_sh", False))
    REGISTRY.register(
        "sh_to_rgb_deg1", _sh_to_rgb_deg1,
        enabled=compile_sh,
    )
    REGISTRY.register(
        "sh_to_rgb_deg2", _sh_to_rgb_deg2,
        enabled=compile_sh,
    )
    REGISTRY.register(
        "sh_to_rgb_deg3", _sh_to_rgb_deg3,
        enabled=compile_sh,
    )
    # Tiny helpers — default-off, opt-in for ablation
    REGISTRY.register(
        "effective_count_core", _effective_count_core,
        enabled=bool(getattr(args, "compile_energy", False)),
    )
    REGISTRY.register(
        "effective_count_loss_core", _effective_count_loss_core,
        enabled=bool(getattr(args, "compile_energy", False)),
    )
    REGISTRY.register(
        "opacity_entropy_core", _opacity_entropy_core,
        enabled=bool(getattr(args, "compile_energy", False)),
    )
    REGISTRY.register(
        "utility_core", _utility_core,
        enabled=bool(getattr(args, "compile_utility", False)),
    )

    global _COMPILE_CONFIG
    mode = getattr(args, "compile_mode", "off")
    _COMPILE_CONFIG = {
        "requested_compile_mode": str(mode),
        "resolved_compile_mode": str(mode),
        "compile_dynamic_requested": bool(getattr(args, "compile_dynamic", True)),
        "compile_dynamic_resolved": bool(getattr(args, "compile_dynamic", True)),
        "compile_after_iter": int(getattr(args, "compile_after_iter", 12000)),
        "torch_compile_available": bool(_COMPILE_AVAILABLE),
        "is_rocm": bool(_IS_ROCM),
        "compile_sh": compile_sh,
        "compile_energy": bool(getattr(args, "compile_energy", False)),
        "compile_utility": bool(getattr(args, "compile_utility", False)),
        # Adjacent backend choices that also govern throughput (#597): recorded
        # here so `compile_mode=off` is never read as "all optimized kernels
        # off". Read from the SAME namespace both trainers hand to this
        # function, so neither trainer needs its own wiring.
        # Read WITHOUT default substitution -- see _observed().
        "ssim_backend_requested": _requested_ssim_backend(args),
        "optimizer_type": (
            None if _observed(args, "optimizer_type") is None
            else str(_observed(args, "optimizer_type")).lower()
        ),
        "gsplat_sparse_grad": (
            None if _observed(args, "gsplat_sparse_grad") is None
            else bool(_observed(args, "gsplat_sparse_grad"))
        ),
        "sh_update_interval": (
            None if _observed(args, "sh_update_interval") is None
            else int(_observed(args, "sh_update_interval") or 1)
        ),
        "parallelism_profile": (
            None if _observed(args, "parallelism_profile") is None
            else str(_observed(args, "parallelism_profile"))
        ),
        "sh_impl": (
            None if _observed(args, "sh_impl") is None
            else str(_observed(args, "sh_impl"))
        ),
    }
    if mode == "off":
        REGISTRY.configure(enabled=False)
        _COMPILE_CONFIG["resolved_compile_mode"] = "off"
        _COMPILE_CONFIG["compile_after_iter"] = None
        print("[compiled-kernels] torch.compile disabled (--compile_mode=off)")
        print(format_backend_summary(build_compute_backend_block()), flush=True)
        return

    # ROCm warning
    if _IS_ROCM:
        print(
            "[compiled-kernels] WARNING: torch.compile on ROCm is experimental "
            "for this workload and may be slower. Use for ablation only.",
            flush=True,
        )

    # Dynamo/Inductor cache limits to prevent unbounded graph growth
    try:
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
        torch._dynamo.config.cache_size_limit = 16
        torch._dynamo.config.accumulated_cache_size_limit = 64
    except Exception:
        pass

    dynamic = bool(getattr(args, "compile_dynamic", True))
    # MCMC changes the Gaussian count N every step. With reduce-overhead (CUDA
    # graphs) a static (dynamic=False) compile recompiles on every shape change —
    # observed as repeated ~13s stalls that idle the GPU on live runs. Force
    # dynamic shapes for this mode so shape changes don't retrigger compilation.
    if mode == "reduce-overhead" and not dynamic:
        print(
            "[compiled-kernels] forcing dynamic=True for reduce-overhead "
            "(streaming/MCMC changes N every step; static compile recompiles repeatedly)",
            flush=True,
        )
        dynamic = True
    after_iter = int(getattr(args, "compile_after_iter", 12000))
    REGISTRY.configure(enabled=True, mode=mode, dynamic=dynamic, compile_after_iter=after_iter)
    _COMPILE_CONFIG["compile_dynamic_resolved"] = bool(dynamic)
    _COMPILE_CONFIG["compile_after_iter"] = after_iter
    print(
        f"[compiled-kernels] mode={mode} dynamic={dynamic} after_iter={after_iter} "
        f"rocm={_IS_ROCM}"
    )
    print(format_backend_summary(build_compute_backend_block()), flush=True)


def set_compile_iteration(iteration):
    """Call every iteration so the registry knows when to switch eager -> compiled."""
    REGISTRY.set_iteration(iteration)
