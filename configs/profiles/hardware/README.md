# Hardware execution profiles

These profiles capture the true-camera kernel knees measured at core source
`57fd2608b8c6c37f5804ac704a3d631edb9e57fd` in
`/tmp/eligsir-qualification/exact57fd2608-v1`
for 9,506 Gaussians,
RGB+ED, SH3, forward+backward, packed+sparse. They are kernel-derived
execution policies, not end-to-end mapper or reconstruction-quality claims.

The profiles intentionally do not set `streaming_optimization_batch_size`
(method batch B). They only cap renderer camera concurrency C and optionally
set an end-to-end optimization-transaction freshness budget. Runtime always
enforces `C <= B` and further reduces C when fewer eligible views exist.

Modes:

- `live_throughput`: measured live throughput knee under a 25 ms p95 target.
- `high_frequency`: measured C with more than 100 renderer transactions/s,
  using a 10 ms freshness target. Full mapper updates/s will be lower.
- `refinement`: throughput knee for source-complete refinement; no explicit
  live deadline is declared by the profile.

The superseded failed R9700 run and one-point plot are not profile evidence.
Requalify these policies end-to-end as map population, resolution, gsplat,
driver, or hardware changes.
