# Curated paper configurations

These files are self-contained publication overrides. They contain only keys accepted by the retained CLI/config schema; dataset locations are not embedded in an anonymous snapshot and must be supplied with `-s/--source_path`.

The recipes are configuration starting points, not validated paper-result
claims. A complete comparison still requires registered dataset, pose, runtime,
and evaluation provenance under the final publication protocol.

Example:

```bash
python train.py -s /data/TUM/fr3_office -m output/fr3 --config configs/paper/tum_fr3.yaml
```

Each file explicitly records dataset, causal trainer, replay, adaptive-supervision, lazy-densification/topology, and evaluation choices. Replica sequence selection is represented by the source directory because the current CLI has no separate Replica sequence key. Every recipe is marked `NOT_YET_PAPER_GRADE`: exact final paper values and scientific validation are not established by this snapshot.
