from __future__ import annotations

import json
from types import SimpleNamespace

from eligsir.experiments.hashing import content_hash
from eligsir.experiments.w3_02_runtime import W3_02OptimizationTimingRecorder


class _Event:
    def __init__(self, elapsed_ms=0.0):
        self.elapsed_ms = elapsed_ms
        self.recorded = False

    def record(self):
        self.recorded = True

    def elapsed_time(self, _other):
        return self.elapsed_ms


class _Cuda:
    def __init__(self):
        self.events = [_Event(2.5), _Event(0.0)]
        self.synchronized = 0

    def is_available(self):
        return True

    def Event(self, enable_timing):
        return self.events.pop(0)

    def synchronize(self):
        self.synchronized += 1


def test_recorder_writes_gpu_event_rows_and_binds_runtime_inputs(tmp_path):
    manifest = {"experiment_id": "w3-02", "config_hash": "a" * 64}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    checkpoint = tmp_path / "initial.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    mutation_trace = tmp_path / "trace.jsonl"
    mutation_trace.write_bytes(b"trace")
    cfg_args = tmp_path / "cfg_args"
    cfg_args.write_text("Namespace(seed=7)")
    reproduce = tmp_path / "reproduce.sh"
    reproduce.write_text("#!/bin/sh\n")
    psnr_history = tmp_path / "psnr_history.jsonl"
    psnr_history.write_text('{"iteration": 100, "mean_test_psnr": 22.0}\n')
    report_dir = tmp_path / "comparison" / "iter_100"
    report_dir.mkdir(parents=True)
    (report_dir / "report.json").write_text('{"eval_valid": true, "n_test": 1}')
    frozen = {}
    for name in ("config_hash", "source_artifact_hash", "pose_hash", "keyframe_hash", "holdout_hash"):
        path = tmp_path / f"{name}.json"
        path.write_text(name)
        frozen[name] = path
    torch = SimpleNamespace(cuda=_Cuda())
    recorder = W3_02OptimizationTimingRecorder(torch)

    recorder.begin(100)
    recorder.end()
    output = recorder.finalize(
        model_path=tmp_path,
        manifest_path=manifest_path,
        arm="baseline",
        git_commit="b" * 40,
        clean_checkout=True,
        seed=7,
        checkpoint_path=checkpoint,
        mutation_trace_path=mutation_trace,
        frozen_artifact_paths=frozen,
    )

    assert output == tmp_path / "w3_02_runtime_evidence.json"
    payload = json.loads(output.read_text())
    assert payload["manifest_hash"] == content_hash(manifest)
    assert payload["optimization_gpu_events"][0] == {"iteration": 100, "gpu_ms": 2.5}
    assert payload["runtime"]["initial_checkpoint_hash"]
    assert payload["runtime"]["mutation_trace_hash"]
    assert payload["runtime"]["mutation_trace_applied"] is False
    assert payload["runtime"]["frozen_artifacts"]["source_artifact_hash"]["path"] == str(frozen["source_artifact_hash"])
    assert payload["runtime"]["artifact_hashes"]["final_report"]
    assert torch.cuda.synchronized == 1
