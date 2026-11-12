# External evaluation assets

EliGSiR includes the LPIPS implementation and a pinned manifest, but does not
redistribute LPIPS calibration checkpoints or ImageNet backbone checkpoints.
Their upstream redistribution terms are not resolved. The manifest retains the
exact URL, filename, byte count, and SHA-256 for each required asset so a user
with the appropriate upstream rights can obtain and verify them.

The Dockerfiles and normal runtime never download these files. To prepare a
private evaluation directory explicitly:

```bash
python scripts/materialize_eval_assets.py \
  --manifest lpipsPyTorch/eval_assets.json \
  --destination "$PWD/eval-assets"
```

Mount that host directory at `/opt/eligsir/eval-assets`, or set
`ELIGSIR_EVAL_ASSET_DIR` to the mounted location. The LPIPS loader verifies
the manifest entry before reading a checkpoint and fails closed when the file
is absent or does not match its recorded size or digest.

The materialized files remain external user-owned inputs. Running the helper
does not grant permission to redistribute them, add them to a GitHub Release,
or bake them into a Docker image.
