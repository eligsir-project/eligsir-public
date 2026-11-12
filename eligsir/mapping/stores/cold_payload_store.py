"""Explicit bounded ownership for non-reloadable cold CPU payloads."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
from tempfile import mkdtemp
from typing import Any, Generic, Optional, TypeVar

import numpy as np

from eligsir.contracts.runtime import PixelReloadToken


PayloadT = TypeVar("PayloadT")


@dataclass(frozen=True, slots=True)
class EvictedColdPayload(Generic[PayloadT]):
    """One payload removed while enforcing the configured resident bounds."""

    frame_id: int
    payload: PayloadT
    payload_bytes: int


class BoundedColdPayloadStore(Generic[PayloadT]):
    """LRU owner bounded independently by resident frame count and bytes.

    ``None`` preserves the legacy unbounded behavior for callers that have not
    yet opted into live retention limits.  A configured zero is a real bound
    and therefore retains no resident payloads.
    """

    def __init__(
        self,
        *,
        max_frames: Optional[int] = None,
        max_bytes: Optional[int] = None,
    ) -> None:
        if max_frames is not None and int(max_frames) < 0:
            raise ValueError("max cold payload frames must be >= 0")
        if max_bytes is not None and int(max_bytes) < 0:
            raise ValueError("max cold payload bytes must be >= 0")
        self.max_frames = None if max_frames is None else int(max_frames)
        self.max_bytes = None if max_bytes is None else int(max_bytes)
        self._entries: "OrderedDict[int, tuple[PayloadT, int]]" = OrderedDict()
        self._total_bytes = 0

    def put(
        self,
        frame_id: int,
        payload: PayloadT,
        *,
        payload_bytes: int,
        evict: bool = True,
    ) -> tuple[EvictedColdPayload[PayloadT], ...]:
        size = int(payload_bytes)
        if size < 0:
            raise ValueError("cold payload size must be >= 0")
        key = int(frame_id)
        prior = self._entries.pop(key, None)
        if prior is not None:
            self._total_bytes -= prior[1]
        self._entries[key] = (payload, size)
        self._total_bytes += size

        evicted = []
        while evict and self._over_limit():
            evicted_id, (evicted_payload, evicted_bytes) = self._entries.popitem(
                last=False
            )
            self._total_bytes -= evicted_bytes
            evicted.append(
                EvictedColdPayload(
                    frame_id=evicted_id,
                    payload=evicted_payload,
                    payload_bytes=evicted_bytes,
                )
            )
        return tuple(evicted)

    @property
    def over_limit(self) -> bool:
        """Whether resident payloads exceed either configured bound."""
        return self._over_limit()

    @property
    def eviction_ids(self) -> tuple[int, ...]:
        """Return the minimal LRU prefix needed to restore the resident bound."""
        total_bytes = self._total_bytes
        entries = list(self._entries.items())
        index = 0
        while (
            (self.max_frames is not None and len(entries) - index > self.max_frames)
            or (self.max_bytes is not None and total_bytes > self.max_bytes)
        ):
            _frame_id, (_payload, payload_bytes) = entries[index]
            total_bytes -= payload_bytes
            index += 1
        return tuple(frame_id for frame_id, _entry in entries[:index])

    def eviction_ids_for_incoming(self, *, payload_bytes: int) -> tuple[int, ...]:
        """Return the bounded LRU prefix needed before one new payload enters.

        The caller owns the prospective payload and must not put it until this
        prefix has been persisted.  A payload that cannot fit even after every
        resident victim is released is rejected rather than briefly exceeding
        the configured RAM bound.
        """

        size = int(payload_bytes)
        if size < 0:
            raise ValueError("cold payload size must be >= 0")
        total_bytes = self._total_bytes + size
        entries = list(self._entries.items())
        index = 0
        while (
            (self.max_frames is not None and len(entries) - index + 1 > self.max_frames)
            or (self.max_bytes is not None and total_bytes > self.max_bytes)
        ):
            if index == len(entries):
                raise ValueError("cold payload cannot fit configured resident bounds")
            _frame_id, (_payload, victim_bytes) = entries[index]
            total_bytes -= victim_bytes
            index += 1
        return tuple(frame_id for frame_id, _entry in entries[:index])

    def can_put_without_eviction(self, *, payload_bytes: int) -> bool:
        """Whether one new resident payload fits without releasing any entry."""
        size = int(payload_bytes)
        if size < 0:
            raise ValueError("cold payload size must be >= 0")
        return not (
            (self.max_frames is not None and len(self._entries) + 1 > self.max_frames)
            or (self.max_bytes is not None and self._total_bytes + size > self.max_bytes)
        )

    def _over_limit(self) -> bool:
        return bool(
            (self.max_frames is not None and len(self._entries) > self.max_frames)
            or (self.max_bytes is not None and self._total_bytes > self.max_bytes)
        )

    def get(self, frame_id: int, *, touch: bool = True) -> Optional[PayloadT]:
        entry = self._entries.get(int(frame_id))
        if entry is None:
            return None
        if touch:
            self._entries.move_to_end(int(frame_id))
        return entry[0]

    def remove(self, frame_id: int) -> Optional[PayloadT]:
        entry = self._entries.pop(int(frame_id), None)
        if entry is None:
            return None
        self._total_bytes -= entry[1]
        return entry[0]

    def contains(self, frame_id: int) -> bool:
        return int(frame_id) in self._entries

    @property
    def frame_ids(self) -> tuple[int, ...]:
        return tuple(self._entries)

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    def __len__(self) -> int:
        return len(self._entries)


@dataclass(frozen=True, slots=True)
class ColdPayloadSpillRequest:
    """One immutable CPU payload selected by the COLD owner for worker I/O."""

    frame_id: int
    source_identity: str
    payload: Any


@dataclass(frozen=True, slots=True)
class ColdPayloadSpillReceipt:
    """Worker result that the mapping owner may publish without disk I/O."""

    frame_id: int
    source_identity: str
    token: PixelReloadToken | None
    error: str | None = None


class ColdPayloadDiskCache:
    """Lossless local backing owned by the bounded COLD payload store."""

    PROVIDER = "prepared_cold_disk_rgbd_v1"

    def __init__(self, root: str | Path, *, max_bytes: int | None = None) -> None:
        if max_bytes is not None and int(max_bytes) < 0:
            raise ValueError("cold payload disk bytes must be >= 0")
        self.root = Path(root)
        self.max_bytes = None if max_bytes is None else int(max_bytes)
        self._stored_bytes = 0

    @staticmethod
    def _array(value: object, name: str) -> np.ndarray:
        if not isinstance(value, np.ndarray):
            raise TypeError(f"cold payload spill requires a CPU ndarray for {name}")
        return value

    def write(self, request: ColdPayloadSpillRequest) -> ColdPayloadSpillReceipt:
        """Atomically write one spill record; report failure for owner handling."""
        temporary: Path | None = None
        try:
            payload = request.payload
            rgb = self._array(getattr(payload, "rgb", None), "rgb")
            depth = getattr(payload, "depth", None)
            if depth is not None:
                depth = self._array(depth, "depth")
            payload_bytes = int(rgb.nbytes) + (0 if depth is None else int(depth.nbytes))
            if self.max_bytes is not None and self._stored_bytes + payload_bytes > self.max_bytes:
                raise OSError("cold payload disk capacity exhausted")
            self.root.mkdir(parents=True, exist_ok=True)
            target = self.root / f"{request.frame_id}-{request.source_identity}"
            if target.exists():
                raise FileExistsError("cold payload spill target already exists")
            temporary = Path(mkdtemp(prefix=".spill-", dir=self.root))
            rgb_path = temporary / "rgb.npy"
            depth_path = temporary / "depth.npy"
            np.save(rgb_path, rgb, allow_pickle=False)
            if depth is not None:
                np.save(depth_path, depth, allow_pickle=False)
            (temporary / "manifest.json").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "source_identity": request.source_identity,
                        "frame_id": int(request.frame_id),
                        "rgb": {"shape": list(rgb.shape), "dtype": str(rgb.dtype), "bytes": int(rgb.nbytes)},
                        "depth": None if depth is None else {"shape": list(depth.shape), "dtype": str(depth.dtype), "bytes": int(depth.nbytes)},
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            os.replace(temporary, target)
            self._stored_bytes += payload_bytes
            locators = (str(target / "manifest.json"), str(target / "rgb.npy"))
            if depth is not None:
                locators += (str(target / "depth.npy"),)
            return ColdPayloadSpillReceipt(
                request.frame_id,
                request.source_identity,
                PixelReloadToken(
                    provider=self.PROVIDER,
                    locators=locators,
                    parameters=(("frame_id", str(request.frame_id)), ("source_identity", request.source_identity)),
                ),
            )
        except Exception as error:  # worker reports storage failure to the owner
            if temporary is not None:
                shutil.rmtree(temporary, ignore_errors=True)
            return ColdPayloadSpillReceipt(
                request.frame_id,
                request.source_identity,
                None,
                f"{type(error).__name__}: {error}",
            )

    def load(self, token: PixelReloadToken) -> tuple[np.ndarray, np.ndarray | None]:
        """Load and cheaply validate a cache-owned token on the prefetch worker."""
        if token.provider != self.PROVIDER or len(token.locators) not in {2, 3}:
            raise ValueError("unsupported prepared cold disk token")
        paths = tuple(Path(path).resolve() for path in token.locators)
        root = self.root.resolve()
        if any(root not in path.parents for path in paths):
            raise ValueError("prepared cold disk token escapes its cache root")
        manifest = json.loads(paths[0].read_text(encoding="utf-8"))
        parameters = dict(token.parameters)
        if (
            manifest.get("version") != 1
            or manifest.get("source_identity") != parameters.get("source_identity")
            or int(manifest.get("frame_id", -1)) != int(parameters.get("frame_id", -1))
        ):
            raise ValueError("prepared cold disk manifest identity mismatch")
        rgb = np.load(paths[1], allow_pickle=False)
        depth = None if len(paths) == 2 else np.load(paths[2], allow_pickle=False)
        for label, value, saved in (("rgb", rgb, manifest["rgb"]), ("depth", depth, manifest["depth"])):
            if saved is None:
                if value is not None:
                    raise ValueError("prepared cold disk manifest depth mismatch")
                continue
            if value is None or list(value.shape) != saved["shape"] or str(value.dtype) != saved["dtype"] or int(value.nbytes) != int(saved["bytes"]):
                raise ValueError(f"prepared cold disk {label} layout mismatch")
        return rgb, depth
