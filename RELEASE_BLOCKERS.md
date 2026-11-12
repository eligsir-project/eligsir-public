# Release blockers

This is a private, incomplete publication checkpoint. The source documentation and
static release preflight have been improved, but no public-release approval is implied.
The resolved software
and export-sanitization items below do not constitute public-release readiness.

## Resolved checkpoint gates

- [x] Runtime utility closure: the curated export retains 98 runtime utility
  modules, including `eligsir.utils.admission_fingerprint`. The acceptance
  surface records 193 passing CPU tests for this closure. This is software
  acceptance evidence, not end-to-end mapping or scientific evidence.
- [x] Public workflow replacement: the export retains a read-only,
  GitHub-hosted `ubuntu-latest` validation workflow with no self-hosted runner,
  internal registry, private token, or publish step.
- [x] Publication residue and scope: the private Ablator declaration is
  quarantined; image/base references are caller-supplied; legacy image names,
  private dependency URLs, and the `uv.lock` project-name mismatch are removed.
  The docs checker is scope-aware for the intentionally pruned corpus while
  retaining checks over every retained document.
- [x] Native/LPIPS provenance mechanics: `fused-ssim` is pinned to an exact
  commit with its MIT notice; this establishes publication reproducibility but
  does not reconstruct historical build provenance. The gsplat ROCm fork has
  its neutral publication repository (private until coordinated release), exact
  detached SHA, upstream base, Apache-2.0 license, AMD notice, and
  publication-rights evidence recorded. The LPIPS implementation and
  calibration attribution are mapped and noticed; no model weights are bundled.

## Open release gates

- [ ] Legal and provenance decisions remain open for the retained or declared
  third-party components:
  - LPIPS implementation and calibration attribution are resolved. Only the
    redistribution terms for the PyTorch-hosted AlexNet/VGG/SqueezeNet
    ImageNet pretrained weights remain open; no weights are bundled.
  - ORB-SLAM3 and the ROS 2 wrapper are gitignored/unshipped (zero tracked
    source files), so their exact GPL/source pin is an external integration and
    container-qualification gate, not a claim about bundled source. Retained
    build/docs references still imply optional integration, so distribution
    remains blocked until exact pins/notices and a distribution decision exist.
  - RVizSplat/`gsplat_msgs` has no authoritative retrievable license/notice;
    retain an authoritative source snapshot or remove the optional viewer
    distribution.
  - TUM RGB-D, Replica, ScanNet/ScanNet++, and Hypersim remain external data
    with distinct terms; record current attribution, access agreements, and
    version provenance. Do not bundle the datasets.
  - The retained GraphDeco-derived code remains governed by its upstream
    research/evaluation and non-commercial terms. EliGSiR-original code is
    separately available under Apache-2.0; neither grant changes the terms of
    the unresolved third-party components below.
- [x] Hosted CI execution: the publication contract was exercised on the
  committed RC at [`2f1f1b421e6deb6e8bdc6ad1c44063dc5c577f07`](https://github.com/eligsir-project/eligsir/commit/2f1f1b421e6deb6e8bdc6ad1c44063dc5c577f07).
  The GitHub-hosted validation runs [34837898050](https://github.com/eligsir-project/eligsir/actions/runs/34837898050)
  and [34838018988](https://github.com/eligsir-project/eligsir/actions/runs/34838018988)
  completed successfully. This evidence covers the publication contract and
  hosted CPU/static validation; it is distinct from the broad legacy suite
  below.
- [ ] The latest CPU diagnostic still has 10 failures and 30 collection errors
  (with 1,154 passing tests). Many collection errors concern pruned legacy
  test dependencies; others are native GPU-library imports. The full suite
  must not be reported as clean.
- [ ] No end-to-end GPU, camera, container-runtime, or other hardware
  qualification has been completed for this export. No scientific result or
  public deployment claim is implied.

See `docs/THIRD_PARTY_PROVENANCE.yaml`, `THIRD_PARTY_NOTICES.md`, and
`docs/PROVENANCE_MATRIX.md` for the component-level evidence and unresolved
terms. This file intentionally records blockers rather than granting release
permission.

## Identity and publication-surface gates (added 12 November 2026)

- [ ] **Do not switch this existing development repository to public** for a double-anonymous release. Its local `main` commit author fields are neutral, but some feature-branch commits and GitHub-hosted PR/Actions actor records are personally attributed. Make a fresh repository from a clean file export; do not mirror, fork or transfer the existing Git history and platform activity.
- [ ] Run `scripts/audit_release_identity.py` on the exported tree and the strict history/ref check on the new release repository. Also independently inspect GitHub platform actors, binary metadata and a full-history secret scan; local code checks cannot cover these surfaces.
- [ ] Supply a **publicly retrievable** pinned ROCm gsplat fork and immutable base image, or clearly restrict the release to a separately validated backend. The recorded fork pin/notice alone does not establish publicly executable reproduction.
- [ ] Complete clean-host mapping → rendering → exact-metric GPU reproduction and compare paper-matched quality-versus-time curves. The `configs/paper/` recipes remain `NOT_YET_PAPER_GRADE` until this evidence exists.
- [ ] Verify that public paper/project links and contributor metadata comply with any still-active double-anonymous submissions. An anonymous Git history does not anonymize a linked author-identified preprint.

See [the release checklist](docs/RELEASE_CHECKLIST.md). The static preflight and README fixes do not close these evidence gates.
