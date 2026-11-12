# Copy/paste run templates

These files are intentionally small templates for launching EliGSiR without
reconstructing long command lines by hand.

The pattern is:

1. choose a **run profile** for the execution semantics;
2. put run-specific values in a small YAML **config overlay**;
3. keep the command/spec itself minimal.

EliGSiR merges `configs/profiles/default.yml`, the selected
`--run_profile`, then `--config`; explicit CLI flags win over both.

## 1. Normal pipeline run

Edit the two `CHANGE ME` paths in `normal/config.yml`, then run:

```bash
bash examples/runs/normal/run.sh
```

The template uses the canonical TUM RGB-D causal-replay profile:

```text
causal_replay/tum_rgbd
```

To run another mode, change only `PROFILE` in `normal/run.sh` (for example a
live profile under `configs/profiles/live/`) and adjust the overlay keys.

## 2. One run through Ablator

Edit the scene path in `ablator/single.json` and any experiment settings in
`ablator/config.yml`, then enqueue it:

```bash
bash examples/runs/ablator/run_single.sh
```

This creates one Ablator job with one arm. The Ablator host config remains the
machine-local deployment config (`~/.config/ablator/config.toml` by default);
the EliGSiR training settings stay in the tracked YAML overlay.

To execute one queued job on the current machine:

```bash
ablator run --once
```

For normal runner daemons/multi-machine operation, use the usual `ablator run`
or `ablator start` workflow instead.

## 3. Multi-arm ablation plan

Edit the scene path in `ablation/plan.json`. The example compares a 500k
Gaussian control with a 250k capacity arm. The only arm-specific difference is
kept in YAML:

- `ablation/configs/control.yml`
- `ablation/configs/cap_250k.yml`

Both inherit `ablation/configs/common.yml`.

Enqueue the plan with:

```bash
bash examples/runs/ablation/plan.sh
```

The plan is sequential by default (`"parallel": false`) so the two arms cannot
compete for the same GPU. Change it to `true` only when the selected machines
provide independent resources.

## Copying a template

Prefer copying the whole relevant directory, then changing:

- `scene` / `source_path`
- `model_path` for direct runs
- the profile name
- the experiment-specific YAML keys
- `machine` in an Ablator spec when a run must target a specific worker

Do **not** copy the large `configs/ablator.toml` into individual experiments.
That file configures the dispatcher/machines. Experiment semantics belong in a
small EliGSiR YAML config and an Ablator run/plan spec.
