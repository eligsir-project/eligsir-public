# EliGSiR source-publication checklist

This file defines a **release gate**, not evidence that a release has been approved. Keep the existing development repository **private** until the relevant academic review, repository-anonymity and rights decisions are complete.

## 1. Freeze a reviewable source candidate

- [ ] Select and record an immutable source commit and list intentionally included features.
- [ ] Exclude unrelated experimental branches, local datasets, checkpoints, results, logs and machine-specific caches.
- [ ] Preserve file-level upstream notices and the provenance matrix when preparing the export.
- [ ] Decide whether the destination is an **anonymous review artifact** or an author-identified archival release. A public preprint/project page may reveal the researchers even if the Git author field is neutral.

## 2. Anonymity and sensitive-data preflight

- [ ] Run `python scripts/audit_release_identity.py` on the exact exported tree.
- [ ] On the **destination snapshot repository**, run `python scripts/audit_release_identity.py --require-anonymous-history --all-refs`. Every reachable commit/ref must be acceptable for the chosen anonymity policy.
- [ ] Inspect image, video and document metadata and visually check any screenshots, labels, slide assets, paths and logos.
- [ ] Run a dedicated secret scanner (such as Gitleaks) against the **entire destination history**; the built-in checker detects only selected high-confidence patterns.
- [ ] Check **GitHub-hosted metadata**, which local Git cannot sanitize: commit associations, PR/issue authors, Actions actors/logs, releases, discussions, pages, organization members and repository events. Existing private development activity must not be inherited through a fork, mirror or visibility switch.
- [ ] Verify that any linked paper/project page or third-party repo does not undermine the particular double-anonymous submission rules that still apply.

**Use a fresh repository created from an exported file snapshot**, with no inherited development refs, PRs, Actions, issues or events. Rewriting Git author names alone does not hide the identity of an existing PR or Actions actor. Do not assume that deleting GitHub objects erases all platform-visible traces.

## 3. Build and contract gate

- [ ] The public-source wheel builds and entry points execute in a clean Python 3.10+ environment.
- [ ] The release CI is green at the exact candidate SHA.
- [ ] `python scripts/check_publication_provenance.py` and the identity preflight pass.
- [ ] Investigate the latest broad CPU diagnostic (**10 failures and 30 collection errors**, alongside 1,154 passes); report the actual executed suite, failures, and exclusions rather than upgrading the narrow CI pass into a full-suite claim.
- [ ] Validate a minimal end-to-end CUDA or ROCm RGB-D mapping run, including checkpoint load, render and metrics, on a clean host.
- [ ] Any ROCm gsplat fork and container base required by the published recipe are **publicly available** at immutable pins; the public build does not need private credentials.

## 4. Scientific reproducibility

- [ ] Freeze dataset name/version, permitted access, RGB-D preprocessing, pose source, train/eval split, config, seed, hardware and dependencies.
- [ ] Run and record at least one clean-host TUM and Replica mapping–render–score reproduction.
- [ ] Provide paper-matched final metrics plus quality-versus-wall-clock curves and time-to-quality thresholds under paired compute budgets.
- [ ] Identify differences between development example configs and actual paper configs. Until qualified, keep `NOT_YET_PAPER_GRADE` labels and avoid reproduced-results claims.

## 5. Redistribution rights and announcement

- [ ] Reconcile `docs/THIRD_PARTY_PROVENANCE.yaml` with the exact shipped source and container images.
- [ ] Keep all third-party assets/weights/datasets external unless redistribution rights are documented; explicitly remove unused optional integrations if the license/pin cannot be established.
- [ ] Retain the distinct GraphDeco non-commercial terms and EliGSiR-original Apache-2.0 boundary.
- [ ] Obtain the required project/institutional release authorization and determine whether other active review tracks permit the intended public links.
- [ ] Replace placeholders, verify URLs, add approved citation metadata, and announce the code only after all relevant blocking gates pass.

For current findings see [RELEASE_BLOCKERS.md](../RELEASE_BLOCKERS.md). No checklist item is automatically satisfied by a passing provenance test.
