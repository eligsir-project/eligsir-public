# Real `pair accounting:` log fixtures (issue #1114)

These are **verbatim** lines captured from a real `eligsir_rgbd_sync`
`sync_node` process (image `localhost/rgbd-sync:i865`, built from
`~/git/eligsir-rgbd-sync` at `1cbc33f`), not hand-written from reading
`pair_ledger.cpp`. That distinction matters: `utils/sync_pair_ledger.py`'s
parser is the load-bearing half of #1114's fix, and a regex tested only
against a format string transcribed from the emitter proves the parser
matches *our reading* of the emitter, not the emitter.

They retain the ANSI escapes and the `[INFO] [stamp] [node]:` prefix exactly
as `podman logs -f` writes them into `<model_path>/rgbd_sync_node.log`.

Generated 2026-08-23 by running the node inside that image against a tiny
Python publisher that emits 60 color + depth `sensor_msgs/Image` pairs at
~30 Hz with deliberately omitted depth messages:

- `rgbd_sync_node_gap.log` — one mid-stream depth message omitted, so the
  matcher has a real attributed drop. Ends with the `FINAL pair accounting:`
  line the node's destructor prints after `PairLedger::flush()`:
  `color: arrived=60 published=59 pending=0 dropped=1 unpaired_retired=1`,
  `depth: arrived=59 published=59`. The same shape as kitchen1's clean
  replay, in miniature.
- `rgbd_sync_node_pending.log` — four tail depth messages omitted and the
  node stopped before the retire horizon elapsed, so the last periodic
  snapshot carries `pending=4`. This is the case where the *trainer* reads
  the ledger mid-life (the normal case: the report is written while the sync
  node is still running).
