import os
from eligsir.utils.training_curves import plot_training_curves

def test_no_events_returns_none(tmp_path):
    assert plot_training_curves(str(tmp_path)) is None

def test_writes_vector_svg_when_events_exist(tmp_path):
    # minimal TB events with train/psnr
    from torch.utils.tensorboard import SummaryWriter
    w = SummaryWriter(str(tmp_path))
    for i in range(0, 200, 10):
        w.add_scalar("train/psnr", 20.0 if i % 20 == 0 else 8.0, i)  # oscillating
        w.add_scalar("streaming/gaussians", 1000 + i, i)
    w.flush(); w.close()
    out = plot_training_curves(str(tmp_path))
    assert out is not None and os.path.exists(out)
    assert out.endswith("training_curves.svg")
    assert "<svg" in (tmp_path / "training_curves.svg").read_text()
    assert not (tmp_path / "training_curves.png").exists()


def test_seed_fade_ever_active_detects_scalar_present(tmp_path):
    from torch.utils.tensorboard import SummaryWriter
    from eligsir.utils.training_curves import _seed_fade_ever_active

    w = SummaryWriter(str(tmp_path))
    for i in range(0, 100, 10):
        w.add_scalar("train/psnr", 20.0, i)
        w.add_scalar("streaming/seed_fade_active", 1 if i == 50 else 0, i)
    w.flush(); w.close()

    import glob
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    fs = sorted(glob.glob(str(tmp_path / "events.out.tfevents.*")))
    ea = EventAccumulator(fs[-1], size_guidance={"scalars": 0})
    ea.Reload()
    tags = set(ea.Tags().get("scalars", []))

    assert _seed_fade_ever_active(ea, tags) is True


def test_seed_fade_ever_active_false_when_never_set(tmp_path):
    from torch.utils.tensorboard import SummaryWriter
    from eligsir.utils.training_curves import _seed_fade_ever_active

    w = SummaryWriter(str(tmp_path))
    for i in range(0, 100, 10):
        w.add_scalar("train/psnr", 20.0, i)
        w.add_scalar("streaming/seed_fade_active", 0, i)
    w.flush(); w.close()

    import glob
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    fs = sorted(glob.glob(str(tmp_path / "events.out.tfevents.*")))
    ea = EventAccumulator(fs[-1], size_guidance={"scalars": 0})
    ea.Reload()
    tags = set(ea.Tags().get("scalars", []))

    assert _seed_fade_ever_active(ea, tags) is False


def test_seed_fade_ever_active_false_when_tag_absent(tmp_path):
    from torch.utils.tensorboard import SummaryWriter
    from eligsir.utils.training_curves import _seed_fade_ever_active

    w = SummaryWriter(str(tmp_path))
    for i in range(0, 100, 10):
        w.add_scalar("train/psnr", 20.0, i)  # no seed_fade_active tag at all
    w.flush(); w.close()

    import glob
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    fs = sorted(glob.glob(str(tmp_path / "events.out.tfevents.*")))
    ea = EventAccumulator(fs[-1], size_guidance={"scalars": 0})
    ea.Reload()
    tags = set(ea.Tags().get("scalars", []))

    assert _seed_fade_ever_active(ea, tags) is False


def test_plot_training_curves_does_not_crash_with_seed_fade_scalar(tmp_path):
    # Integration-style smoke test: the full plotting pipeline must not
    # error out when the new scalar is present, even though this test
    # doesn't inspect the rendered title text.
    from torch.utils.tensorboard import SummaryWriter
    from eligsir.utils.training_curves import plot_training_curves
    import os

    w = SummaryWriter(str(tmp_path))
    for i in range(0, 200, 10):
        w.add_scalar("train/psnr", 20.0, i)
        w.add_scalar("train/global_psnr", 20.0, i)
        w.add_scalar("streaming/seed_fade_active", 1 if 50 <= i <= 100 else 0, i)
    w.flush(); w.close()
    out = plot_training_curves(str(tmp_path))
    assert out is not None and os.path.exists(out)
