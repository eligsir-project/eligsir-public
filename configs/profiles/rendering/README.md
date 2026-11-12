# Live rasterization A/B profiles

These files are pure renderer overlays and do not select method batch size,
camera microbatch size, admission, replay, resolution, or topology policy.
The CLI accepts one resolved config root, so use a composite profile that
extends the canonical causal mapper, one hardware execution policy, and exactly
one renderer overlay. The qualified local 8060S A/B roots are:

```bash
--config configs/profiles/experiments/live_aa/8060s/classic.yml
--config configs/profiles/experiments/live_aa/8060s/antialiased.yml
```

Those are separate arm examples, not flags to pass together. Do not pass both
`--run_profile` and `--config`; that is intentionally rejected by the loader.
Other hardware can compose the same pure renderer overlays with its measured
profile under `configs/profiles/hardware/` once that A/B is scheduled there.

`classic` remains the default and control arm. `antialiased` enables gsplat's
EWA antialiasing. Compare matched ResearchFlow arms using mapping-active wall
time and supervised views, not artifact iteration alone. Record optimizer
updates/s, supervised views/s, rasterizer GPU time, canonical holdout PSNR,
same-resolution edge PSNR, motion-compensated temporal flicker, Gaussian count,
and every supervision-resolution transition. Missing pose/depth overlap makes
the flicker metric unavailable; it must never silently degrade to a raw
adjacent-frame pixel difference.
