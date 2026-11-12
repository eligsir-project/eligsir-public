# Publication snapshot test policy

The only retained hosted workflow is [`.github/workflows/ci-public.yml`](../.github/workflows/ci-public.yml), running on GitHub-hosted Ubuntu. It validates a Python wheel, the file-level provenance matrix, release-residue and publication-interface contracts, and the tracked-tree identity/credential preflight. It is **not** the original internal `ci-cpu.yml` system described in the old developer checkout.

## Run locally

From a clean checkout with the publication test dependencies installed:

```bash
python scripts/check_publication_provenance.py
python scripts/audit_release_identity.py
python -m pytest -q tests/test_publication_identity.py tests/test_release_preflight.py \
  tests/test_publication_interface.py tests/test_publication_provenance.py \
  tests/test_publication_notices.py tests/test_release_residue.py
```

For an anonymous public **snapshot repository**, also check Git commit metadata and reachable refs:

```bash
python scripts/audit_release_identity.py --require-anonymous-history --all-refs
```

The strict history check is intentionally **not** part of the feature-branch CI: this private development repository has author-identified work branches and PR/Actions actor records. Do not publish it directly while repository-level double-blind anonymity is a requirement. A clean release repository must have its *GitHub* PR, member, Actions, release and workflow metadata reviewed separately; local Git tests cannot validate that hosted metadata.

## Scope and limitations

Passing publication checks establishes packaging, static provenance declarations, and selected CPU contracts. It does not establish a clean full CPU suite, an end-to-end training run, CUDA/ROCm execution, camera integration, public access to private build dependencies, license clearance, or reproduced scientific results. See [release blockers](../RELEASE_BLOCKERS.md) and [the release checklist](../docs/RELEASE_CHECKLIST.md).
