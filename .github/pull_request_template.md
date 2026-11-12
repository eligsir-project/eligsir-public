<!--
Fast-by-default review: prove the claim this PR actually makes. Do not keep a
software implementation draft merely because broader device qualification or
scientific promotion remains pending.

Policy: docs/architecture/2026-08-25-fast-by-default-development-policy.md
-->

## Summary

<!-- What changes, and what software property should be true after merge? -->

## Integration gate

**Merge claim:**
<!-- The deterministic/software contract this PR establishes. -->

**Current gate:** `gate:merge | gate:qualification | gate:science | gate:none`

**Hardware impact:** `NONE | INHERITED | TARGETED | REQUIRED`

<!--
If INHERITED, link the prior qualification and state why the hardware boundary
is unchanged. If REQUIRED, name the exact property that cannot be established
by deterministic/reference software evidence.
-->

**Remaining post-merge qualification:**
<!-- Device/backend/camera/RMW evidence still wanted, or `none`. -->

**Remaining scientific promotion:**
<!-- ResearchFlow experiment/default/paper follow-up, or `none`. -->

- [ ] No unresolved device/scientific follow-up is being presented as a merge blocker unless it is genuinely required to safely enable the changed path.
- [ ] This PR does not claim hardware/scientific properties beyond the evidence below.

## Test plan

<!--
Prefer the cheapest evidence that directly proves the implementation:
static/unit -> deterministic CPU -> CPU-Torch simulation -> targeted device only
when the property is device-dependent.

State which surface actually executed each relevant test. A green torch-free
CPU check is not evidence that a Torch-dependent test ran.
-->

- [ ] Static/unit/contract checks
- [ ] Deterministic CPU or CPU-Torch simulation where applicable
- [ ] Targeted hardware/backend check only if required by the changed property

Commands / tests:

```text
...
```

## Simplification and compaction

<!-- Confirm the passes required by AGENTS.md; this checklist is not a second policy owner. -->

- [ ] Completed the required simplification pass.
- [ ] Completed the required compaction pass without obscuring code or weakening required contracts/test coverage.
- [ ] Re-ran relevant checks after the passes.

## Evidence-bearing claims

<!--
Use this section only when the PR itself makes a reconstruction-quality,
performance, device/backend, scientific, evaluation, or visual-behavior claim.
Implementation PRs may merge without paper-grade evidence when they make no such
claim.

For scientific claims, canonical evidence must be ResearchFlow-managed.
Machine-readable evidence is primary; visuals are supplemental.
-->

- [ ] This PR makes an evidence-bearing claim and machine-readable evidence is linked below.
- [ ] Scientific evidence, if any, is ResearchFlow-managed (or explicitly audited imported evidence).
- [ ] Visual proof is attached when useful, but is not the sole surviving evidence.

Evidence / provenance:

<!-- run/report artifact, command/spec/config, producing commit, dataset/scene/backend identity -->

## Not evidence-bearing

- [ ] This is an implementation/refactor/cleanup/tests/docs/CI change whose merge claim is established by deterministic software evidence; broader qualification/promotion is tracked separately.
