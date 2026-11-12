"""Viewer registry with lazy constructors."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

from eligsir.viewers.sinks.base import PreviewSink
from eligsir.viewers.sinks.viser_sink import ViserUnavailable

from .base import NullViewer, Viewer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ViewerInfo:
    name: str
    status: str
    imports: str
    factory: Callable[..., Viewer]


def _make_null_viewer(**kwargs) -> Viewer:
    return NullViewer()


def _make_rerun_viewer(**kwargs) -> Viewer:
    from eligsir.viewers.rerun_viewer import RerunViewer

    return RerunViewer(**kwargs)


def _build_viser_sink(**kwargs) -> "PreviewSink":
    from eligsir.viewers.sinks.viser_sink import ViserPreviewSink

    port = int(kwargs.get("port", 8080))
    host = str(kwargs.get("host", "0.0.0.0"))
    return ViserPreviewSink(port=port, host=host)


def _make_viser_viewer(**kwargs) -> Viewer:
    """Canonical sidecar viewer path (``--viewer viser`` preset)."""
    return _make_canonical_viewer(("viser",), **kwargs)


_SINK_FACTORIES: Dict[str, Callable[..., "PreviewSink"]] = {
    "viser": _build_viser_sink,
}


def _make_canonical_viewer(sink_names: Tuple[str, ...], **kwargs) -> Viewer:
    """Build a ``CanonicalViewerAdapter`` wired to requested core sinks.

    Adapter-level configuration is consumed here and removed from the kwargs
    passed to sink constructors. Sink factories therefore receive only their
    own transport/process options; adding a new CanonicalViewerConfig field can
    never accidentally become an unexpected keyword argument to a sink.
    """
    from eligsir.viewers.canonical import (
        VIEWER_MAP_MAX_GAUSSIANS_HARD_CEILING,
        CanonicalViewerConfig,
        resolve_viewer_map_max_gaussians,
    )
    from eligsir.viewers.canonical_adapter import CanonicalViewerAdapter

    sink_kwargs = dict(kwargs)
    map_max_gaussians = sink_kwargs.pop("map_max_gaussians", None)
    config = None
    if map_max_gaussians is not None:
        resolved_cap = resolve_viewer_map_max_gaussians(map_max_gaussians)
        if (
            resolved_cap == VIEWER_MAP_MAX_GAUSSIANS_HARD_CEILING
            and int(map_max_gaussians) != VIEWER_MAP_MAX_GAUSSIANS_HARD_CEILING
        ):
            logger.warning(
                "--viewer_map_max_gaussians=%s resolved to the hard safety "
                "ceiling (%d) -- 0 or an over-large value never yields a "
                "truly unbounded live map preview sample.",
                map_max_gaussians,
                VIEWER_MAP_MAX_GAUSSIANS_HARD_CEILING,
            )
        config = CanonicalViewerConfig(map_max_gaussians=resolved_cap)

    sinks = []
    for name in sink_names:
        try:
            factory = _SINK_FACTORIES[name]
        except KeyError as exc:
            raise ValueError(
                f"viewer sink {name!r} has no core factory (known core sinks: "
                f"{', '.join(sorted(_SINK_FACTORIES)) or '<none>'}). Sinks such "
                "as 'rviz' are implemented in an external package (e.g. "
                "eligsir-ros2) and are constructed there against this "
                "core's CanonicalViewer/PreviewSink contract, not selected by "
                "name from this registry."
            ) from exc
        try:
            sinks.append(factory(**dict(sink_kwargs)))
        except ViserUnavailable as exc:
            logger.warning(
                "viewer sink %r failed to start and will be disabled for this "
                "run (training continues without it): %s",
                name,
                exc,
            )
    return CanonicalViewerAdapter(sinks=sinks, config=config)


VIEWERS: Dict[str, ViewerInfo] = {
    "none": ViewerInfo(
        name="none",
        status="stable",
        imports="core",
        factory=_make_null_viewer,
    ),
    "rerun": ViewerInfo(
        name="rerun",
        status="stable",
        imports="optional viewer-rerun",
        factory=_make_rerun_viewer,
    ),
    "viser": ViewerInfo(
        name="viser",
        status="experimental",
        imports="optional viewer-viser (sidecar process only, never imported in the trainer)",
        factory=_make_viser_viewer,
    ),
}


def list_viewers() -> list[str]:
    """Return registered core viewer names."""

    return sorted(VIEWERS)


def make_viewer(name: str = "none", *, viewer_sinks: Optional[str] = None, **kwargs) -> Viewer:
    """Instantiate a registered core viewer by name or explicit sink list."""

    key = str(name or "none")
    if viewer_sinks:
        from eligsir.viewers.canonical import resolve_preset_sinks

        sink_names = resolve_preset_sinks(key, viewer_sinks)
        return _make_canonical_viewer(sink_names, **kwargs)
    try:
        info = VIEWERS[key]
    except KeyError as exc:
        raise ValueError(f"Unknown viewer '{key}'. Choose: {', '.join(list_viewers())}") from exc
    return info.factory(**kwargs)
