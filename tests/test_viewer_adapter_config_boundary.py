from __future__ import annotations

from eligsir.viewers import registry
from eligsir.viewers.sinks.base import PreviewSink


def test_map_cap_is_adapter_config_not_sink_constructor_kwarg(monkeypatch) -> None:
    seen = {}

    def strict_viser_factory(*, port=8080, host="0.0.0.0"):
        seen.update(port=port, host=host)
        return PreviewSink()

    monkeypatch.setitem(registry._SINK_FACTORIES, "viser", strict_viser_factory)
    viewer = registry.make_viewer(
        "viser",
        port=9999,
        host="127.0.0.1",
        map_max_gaussians=500_000,
    )

    assert seen == {"port": 9999, "host": "127.0.0.1"}
    assert viewer.canonical_viewer.config.map_max_gaussians == 500_000
