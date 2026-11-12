# Third-party notices and unresolved provenance

This file is intentionally conservative. It records what is evidenced in the
export and leaves unknown terms unresolved; it does not infer a license from a
package name or from a common implementation pattern.

The file-level project boundary is recorded in
[`docs/RELEASE_RIGHTS.md`](docs/RELEASE_RIGHTS.md): `eligsir-original` files
are available under Apache-2.0, while GraphDeco-derived files retain their
upstream research/evaluation and non-commercial terms. The rows below cover
third-party components and unresolved distribution evidence; they do not
override the file-level matrix or any more specific upstream notice.

| retained path/rule | evidence | current status | release action |
| --- | --- | --- | --- |
| `arguments/`, `scene/`, selected root scripts, and selected `eligsir/utils/` files | Inria/GRAPHDECO headers; matching baseline paths in reference checkout; `LICENSE.md` | Inria/MPII Gaussian-Splatting research/evaluation terms retained | Keep `LICENSE.md` and attribution; do not add a permissive project license |
| `submodules/simple-knn/` | Vendored README names Inria GRAPHDECO and native files retain headers | Inria/MPII terms retained in its `LICENSE.md` | Keep the nested license and attribution |
| `eligsir/utils/loss_utils.py` | Root `LICENSE.md` identifies the pytorch-ssim material by Evan Su | MIT attribution is recorded by the upstream license; exact modified-file scope should be confirmed | Preserve the attribution and verify the source correspondence before release |
| `lpipsPyTorch/` | Retained implementation maps to S-aiueo32 commit `016e0ae19ceff83450590738948941a7eb9b95b7`; calibration assets map to richzhang; exact source URLs and hashes are in the manifest | BSD-2-Clause source/attribution evidence retained; all evaluation-weight terms remain unresolved | No calibration or ImageNet weights are bundled in source releases or Docker images; an entitled user may explicitly materialize them into a host-mounted directory |
| `gaussian_renderer/web_viewer/vendor/` | `chart.umd.min.js` identifies Chart.js 4.4.7 and `playcanvas.mjs` identifies PlayCanvas 2.18.1; hashes and primary sources are recorded in the manifest | MIT notices and exact local hashes are recorded; source/provenance review remains separate | Keep the notices and manifest rows with the bundled files; do not infer SuperSplat scope from the PlayCanvas engine bundle |
| ROCm `gsplat` fork | `Dockerfile` and the ROCm publication compose build contract | Neutral `eligsir-project/gsplat-rocm` source pinned to full commit `c6b363ab9239c958a4ac6619f725119dc8e5706b`; Apache-2.0 LICENSE and AMD MIT NOTICE retained | Keep the exact detached source pin and notice references; this fork blocker is closed for the publication contract; repository remains private until coordinated release |
| `eligsir-orbslam3-src/` | Integration/configuration/build glue refers to an external ORB-SLAM3 source checkout | **UNRESOLVED** | Verify external source license and whether any generated/configured material is redistributable |
| Python/CUDA/ROCm/runtime dependencies declared in packaging or Dockerfiles | Dependency URLs and names are present, but their notices are not vendored in this export | **OUT OF SCOPE / REVIEW REQUIRED** | Build a dependency SBOM/notice bundle for any binary or source distribution |
| datasets and separately distributed ROS2/RGB-D repositories | README and integration paths refer to external datasets/repos | **OUT OF SCOPE / REVIEW REQUIRED** | Review each source repository/dataset terms separately |

No row above grants permission to use trademarks, model weights, datasets, or
external repositories. The unresolved rows are legal release blockers.

## Bundled web assets

The machine-readable records, local hashes, source URLs, and release status
for these files are in
[`docs/THIRD_PARTY_PROVENANCE.yaml`](docs/THIRD_PARTY_PROVENANCE.yaml).
The following notices are retained for the exact bundled files:

### Chart.js 4.4.7 — MIT

Local path: `gaussian_renderer/web_viewer/vendor/chart.umd.min.js`<br>
SHA-256: `206b6e8bb00fc7bba2c7ee80ca41db3e9e05ba7be0aa35abeba9cfd5357f5d0e`<br>
Source: [Chart.js v4.4.7](https://github.com/chartjs/Chart.js/releases/tag/v4.4.7),
[MIT license](https://github.com/chartjs/Chart.js/blob/master/LICENSE.md).<br>
Copyright (c) Chart.js Contributors.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files, to deal in the Software
without restriction, including without limitation the rights to use, copy,
modify, merge, publish, distribute, sublicense, and/or sell copies of the
Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions: the above copyright notice and this
permission notice shall be included in all copies or substantial portions of
the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

### PlayCanvas engine 2.18.1 — MIT

Local path: `gaussian_renderer/web_viewer/vendor/playcanvas.mjs`<br>
SHA-256: `b088507669ba3d3b375fadcc5c626cb4e4fc137dc47a161fbd8144bbb98db311`<br>
Source: [PlayCanvas engine](https://github.com/playcanvas/engine),
[MIT license](https://github.com/playcanvas/engine/blob/main/LICENSE).<br>
Copyright (c) PlayCanvas.

The MIT permission and warranty terms above apply to this PlayCanvas bundle;
the local asset is the PlayCanvas engine bundle, not a claim that the export
contains the separate SuperSplat application.

## Build dependency notices

### fused-ssim — MIT

Retained Docker paths: `Dockerfile`, `Dockerfile.cuda`, and
`Dockerfile.cuda-noble`. Each defaults `FUSED_SSIM_COMMIT` to
`a7c48d6dd7ac6dc39a7958c7c4452e0b10418f38`, shallow-fetches that object, and
checks it out detached before building. This is the chosen publication
reproducibility pin for the audited current `main`, not historical build
provenance. Source: [fused-ssim](https://github.com/rahul-goel/fused-ssim),
[audited commit](https://github.com/rahul-goel/fused-ssim/commit/a7c48d6dd7ac6dc39a7958c7c4452e0b10418f38),
[MIT license](https://github.com/rahul-goel/fused-ssim/blob/main/LICENSE).
Copyright (c) Rahul Goel (2024).

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

### gsplat ROCm fork — Apache-2.0 and AMD MIT notice evidence

The ROCm Docker and compose publication contracts default to the neutral
[`eligsir-project/gsplat-rocm`](https://github.com/eligsir-project/gsplat-rocm.git)
at exact detached commit
[`c6b363ab9239c958a4ac6619f725119dc8e5706b`](https://github.com/eligsir-project/gsplat-rocm/commit/c6b363ab9239c958a4ac6619f725119dc8e5706b).
The pinned fork source retains its Apache-2.0 LICENSE and AMD MIT attribution
in NOTICE; these are the recorded source and rights evidence for this
publication contract. Controlled qualification builds may override
`GSPLAT_REPO` and `GSPLAT_COMMIT`, but must retain a full SHA and detached
checkout. Compare against the upstream
[v1.5.3 license](https://github.com/nerfstudio-project/gsplat/blob/v1.5.3/LICENSE),
[ROCm gsplat source](https://github.com/ROCm/gsplat), and the declared
[publication fork LICENSE](https://github.com/eligsir-project/gsplat-rocm/blob/c6b363ab9239c958a4ac6619f725119dc8e5706b/LICENSE)
and [NOTICE](https://github.com/eligsir-project/gsplat-rocm/blob/c6b363ab9239c958a4ac6619f725119dc8e5706b/NOTICE).
The fork repository remains private until coordinated release; these URLs
identify the intended publication source and do not claim current visibility.

The recorded fork delta is limited to ROCm build integration: honoring
`PYTORCH_ROCM_ARCH`, adding the ROCm GLM include path, restoring the original
GLM tree after hipify, and changing backward rasterization cooperative-group
tiles/reductions from wave64 to wave32. The neutral repository identity, exact
source publication, Apache-2.0 LICENSE, AMD MIT NOTICE, and publication rights
evidence are retained; the gsplat fork blocker is closed for this contract,
while the repository remains private until coordinated release.

### LPIPS implementation — BSD-2-Clause

The retained implementation maps to
[S-aiueo32/lpips-pytorch commit `016e0ae19ceff83450590738948941a7eb9b95b7`](https://github.com/S-aiueo32/lpips-pytorch/commit/016e0ae19ceff83450590738948941a7eb9b95b7),
with the exact downstream GraphDeco reference recorded as commit
`54c035f7834b564019656c3e3fcc3646292f727d` for provenance only. It is not a
claim that the GraphDeco license covers this BSD component. Copyright (c) 2020 Sou Uchida.
Local changes are limited to cache/no-grad behavior,
`weights=None` backbone construction, and fail-closed manifest asset loading.
The implementation hashes and `eval_assets.json` hash are recorded in
[`docs/THIRD_PARTY_PROVENANCE.yaml`](docs/THIRD_PARTY_PROVENANCE.yaml).

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

### LPIPS calibration weights — separate BSD-2-Clause attribution

The LPIPS v0.1 calibration files are separate artifacts from the implementation
and are sourced from
[richzhang/PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity)
under its [BSD-2-Clause license](https://github.com/richzhang/PerceptualSimilarity/blob/master/LICENSE).
Copyright (c) Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman,
and Oliver Wang. The three calibration asset URLs and SHA-256 values are
recorded separately in the manifest. These weights are not bundled in this export or any Docker image. This notice is attribution/provenance, not a claim
about unrelated PyTorch model terms.

### External evaluation assets — explicit user materialization

The manifest at `lpipsPyTorch/eval_assets.json` records the exact source URLs,
sizes, and SHA-256 values for the LPIPS calibration and ImageNet backbone
assets. The publication source and Dockerfiles intentionally contain no `.pth`
files and never download these assets during image builds or normal runtime.
An entitled user who needs LPIPS evaluation may run:

```bash
python scripts/materialize_eval_assets.py \
  --manifest lpipsPyTorch/eval_assets.json \
  --destination "$PWD/eval-assets"
```

Then mount that directory at `/opt/eligsir/eval-assets` (or set
`ELIGSIR_EVAL_ASSET_DIR` to the mounted path). The runtime verifies every
filename, size, and SHA-256 and fails closed when an asset is missing or
corrupt. The command is an explicit user action; it does not resolve the
upstream redistribution terms or authorize redistributing the downloaded files.

### LPIPS backbone weights — unresolved PyTorch terms

`eval_assets.json` records the URLs, sizes, and SHA-256 values for the
PyTorch-hosted AlexNet, VGG16, and SqueezeNet ImageNet backbones required by
the runtime, but no `.pth` files are bundled. Redistribution terms for those
PyTorch-hosted ImageNet weights remain unresolved, so only this narrow weight issue
keeps the LPIPS manifest row release-blocked.

## Components still blocked

These entries have known or reported upstream terms, but remain explicitly
blocked until the missing provenance or rights evidence is retained:

| component | local evidence | known terms | blocking evidence still required |
| --- | --- | --- | --- |
| LPIPS and external evaluation assets | `lpipsPyTorch/`, `lpipsPyTorch/eval_assets.json` | LPIPS implementation and calibration BSD-2-Clause evidence retained; ImageNet terms unresolved | Keep all `.pth` files out of source releases and Docker images; review each model URL/checksum and terms before any redistribution. |
| ORB-SLAM3 and ROS wrapper | `eligsir-orbslam3-src/` (excluded from shipped counts) | ORB-SLAM3 and the Mechazo11 wrapper GPL-3.0 | Exact upstream commits, complete dependency notices, source scope, and distribution decision. |
| RVizSplat/gsplat_msgs | Docker declarations and viewer integration | Unknown | Authoritative license and source snapshot for the declared commit, or remove the optional distribution. |
| External datasets | `configs/paper/`, `eligsir/io/sources/datasets/` | TUM/Replica/ScanNet/Hypersim terms differ; ScanNet++ remains unresolved | Keep data external and record current terms, attribution, access agreement, and version provenance. |

The machine-readable manifest marks every row in this section as
`release_blocked: true`. These notices do not imply commercial/public
permission, blanket relicensing, or rights to model weights, datasets, or
trademarks.
