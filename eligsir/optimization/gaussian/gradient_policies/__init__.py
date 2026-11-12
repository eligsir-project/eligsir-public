"""Gradient-masking policies for the Gaussian optimization-step engine.

Wave 3, PR W3-02 (see
``plans/2026-08-06-wave3-live-mapping-runtime-evidence-strategy-implementation-plan.md``
section 7), Subagent C scope. Each module here extracts one (or a tightly
related group of) gradient-masking block(s) verbatim from the post-backward
cascade in ``train_streaming.py`` (~lines 4532-4700 as of this PR) into a
``GradientPolicyBase`` subclass, per ``eligsir.optimization.gaussian.contracts``'s
``GradientPolicyContext`` design (see that module's docstring for the full
rationale of the five-way grouping used here).
"""
