# EliGSiR

**Streaming RGB-D 3D Gaussian mapping with bounded compute and explicit runtime/evidence contracts.**

[Project page](https://eligsir-project.github.io/) · [Source repository](https://github.com/eligsir-project/eligsir-public)

> **Release candidate, not yet a validated public research reproduction.** The repository is currently a curated source snapshot. Check [release blockers](RELEASE_BLOCKERS.md) and the [release checklist](docs/RELEASE_CHECKLIST.md) before announcing code availability.

## Overview

EliGSiR provides a transport-neutral RGB-D Gaussian-mapping core, with mapping, rendering, evaluation and runtime interfaces. It does not require ROS 2 for the mapping core. ROS 2 deployment, device lifecycle, bag playback, and RViz integration are maintained separately.

## Installation and CPU smoke test

Requires Python 3.10 or newer. In a clean environment:

```bash
python -m pip install -e '.[dev]'
eligsir-sim --backend numpy
python -m eligsir map --help
python -m eligsir render --help
python -m eligsir evaluate --help
```

The simulation/CLI checks do **not** establish GPU mapping or paper-result reproducibility. GPU runs require an appropriate CUDA/ROCm build, native dependencies and a compatible device.

## GPU builds

**CUDA:** The CUDA Dockerfiles pin upstream gsplat commit `937e29912570c372bed6747a5c9bf85fed877bae`. The GPU path must still be independently qualified on a clean host.

**ROCm:** `Dockerfile` and `docker-compose.yml` pin the publication gsplat ROCm fork at `c6b363ab9239c958a4ac6619f725119dc8e5706b`. This source repository is **currently private**, so the ROCm reproduction cannot yet be claimed as publicly executable. Supply a verified digest-pinned `BASE_ROCM_PYTHON` base image to the build; do not substitute unverified tags.

The private release-candidate build accepts an *ephemeral* BuildKit secret rather than embedding a token in an image layer or Dockerfile argument:

```bash
export GITHUB_TOKEN
DOCKER_BUILDKIT=1 docker build \
  --secret id=github_token,env=GITHUB_TOKEN \
  --build-arg BASE_ROCM_PYTHON=registry.example/eligsir-rocm-python@sha256:<digest> \
  --target eligsir -f Dockerfile .
```

This command is for **authorized private builds only** until the fork and base image are publicly accessible. A public build must not require private credentials.

## Datasets

Datasets are **not bundled**. Curated examples reference TUM RGB-D (`fr3_office`, `fr1_desk`) and Replica (`room2`, `office0`). Obtain each dataset from its owner and respect the dataset-specific terms. Supply paths explicitly; no private workstation paths are required or embedded in the curated paper recipes.

## Mapping, rendering, evaluation

The retained development wrappers are:

```bash
bash scripts/reproduce/map_tum_fr3.sh /path/to/TUM/fr3_office output/paper_tum_fr3
bash scripts/reproduce/map_replica_room2.sh /path/to/Replica/room2 output/paper_replica_room2
bash scripts/reproduce/render_comparison.sh output/paper_tum_fr3
bash scripts/reproduce/evaluate.sh output/paper_tum_fr3
```

Run these from the repository root in a **qualified GPU environment** with the appropriate dataset format. The source/package import closure is present and passes its narrow CI contract; these commands are **not yet backed by a clean-host GPU end-to-end reproduction record**. The `configs/paper/` configurations are explicitly `NOT_YET_PAPER_GRADE`. Do not report their output as the paper's frozen benchmark results.

Evaluation uses checkpoint rendering and the project's exact float metric artifacts, not scores recomputed from arbitrary display PNGs. Scientific reproduction additionally requires the matched pose source, dataset version, evaluation holdout, wall-clock definition, hardware, and quality-versus-time reporting.

## Tests and release checks

```bash
python scripts/check_publication_provenance.py
python scripts/audit_release_identity.py
python -m pytest -q tests/test_publication_identity.py tests/test_release_preflight.py \
  tests/test_publication_interface.py tests/test_publication_provenance.py \
  tests/test_publication_notices.py tests/test_release_residue.py
```

The automated checks cover a limited source/packaging contract, not the full CPU suite or GPU hardware qualification. For an **anonymous release candidate** created from a clean repository snapshot, additionally run:

```bash
python scripts/audit_release_identity.py --require-anonymous-history --all-refs
```

That local audit does not inspect GitHub platform actor/PR/Actions metadata. Publishing the existing development repository directly is **not** a validated anonymity strategy; use a new clean repository and verify its remote metadata.

## Code and external components

The core package is in `eligsir/`, dataset/configuration profiles in `configs/`, reproduction wrappers in `scripts/reproduce/`, and publication validation in `tests/`. Related, separately distributed repositories include [eligsir-ros2](https://github.com/eligsir-project/eligsir-ros2) and [eligsir-rgbd-sync](https://github.com/eligsir-project/eligsir-rgbd-sync); their accessibility and licenses are independent of this snapshot.

## Licensing

**There is no blanket Apache-2.0 license on the whole tree.** EliGSiR-original files identified by the [file-level provenance matrix](docs/PROVENANCE_MATRIX.md) carry Apache-2.0. GraphDeco-derived files retain upstream research/evaluation and non-commercial conditions (`LICENSE.md`). Bundled and optional third-party components retain their own notices; consult [license index](LICENSE), [notices](THIRD_PARTY_NOTICES.md), [release rights](docs/RELEASE_RIGHTS.md) and [open blockers](RELEASE_BLOCKERS.md) before redistribution.

