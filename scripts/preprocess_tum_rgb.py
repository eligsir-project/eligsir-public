#!/usr/bin/env python3
"""Condition explicitly allowlisted TRAIN RGB at native resolution in a fresh TUM copy.

The allowlist binds source rgb.txt and production split evidence by SHA256; this
tool does not infer split eligibility. All other files remain byte-identical.
A failed output has no preprocessing_manifest.json and must not be used.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import time

import cv2
import numpy as np


RECEIPT = "preprocessing_manifest.json"
OPERATORS = {
    "identity": {},
    "denoise-only": {"h": 6, "hColor": 6, "templateWindowSize": 7, "searchWindowSize": 21},
    "sharpen-only": {"sigma": 1.2, "strength": 0.15},
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def validate(src, dst, manifest):
    if dst.exists() or dst.is_symlink() or src in dst.resolve().parents or dst.resolve() == src:
        raise ValueError("destination must be fresh and outside source")
    if any((src / name).exists() or (src / name).is_symlink() for name in (RECEIPT, RECEIPT + ".tmp")):
        raise ValueError("source must not contain reserved preprocessing receipt names")
    for path in src.rglob("*"):
        if not path.resolve().is_relative_to(src) or (path.is_symlink() and path.is_dir()):
            raise ValueError(f"outside-source or directory symlink: {path}")
    if not isinstance(manifest, dict) or manifest.get("schema") != "eligsir.train-rgb-allowlist/v1":
        raise ValueError("unsupported allowlist schema")
    for key, identity, checksum in (("source", "scene_id", "rgb_txt_sha256"), ("split", "id", "sha256")):
        obj = manifest.get(key)
        if not isinstance(obj, dict) or not isinstance(obj.get(identity), str) or not obj[identity].strip():
            raise ValueError(f"missing {key} identity")
        if not isinstance(obj.get(checksum), str) or not re.fullmatch("[0-9a-f]{64}", obj[checksum]):
            raise ValueError(f"invalid {key} digest")
    index = (src / "rgb.txt").read_bytes()
    if digest(index) != manifest["source"]["rgb_txt_sha256"]:
        raise ValueError("rgb.txt does not match source identity")
    recognized = set()
    for line in index.decode().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 2:
            raise ValueError("invalid rgb.txt row")
        recognized.add(fields[1])
    allowed = manifest.get("train_rgb_paths")
    if not isinstance(allowed, list) or any(not isinstance(p, str) for p in allowed):
        raise ValueError("train_rgb_paths must be a list of relative RGB paths")
    if len(set(allowed)) != len(allowed):
        raise ValueError("duplicate training RGB path")
    images = {p.relative_to(src).as_posix() for p in (src / "rgb").rglob("*") if p.is_file()}
    if not recognized.issubset(images):
        raise ValueError("rgb.txt references missing or non-RGB source files")
    for rel in allowed:
        path = PurePosixPath(rel)
        if (path.is_absolute() or ".." in path.parts or "\\" in rel or path.as_posix() != rel
                or rel not in recognized or rel not in images or path.suffix.lower() != ".png"):
            raise ValueError(f"unrecognized or unsafe training RGB path: {rel}")
    return set(allowed), sorted(images)


def process_one(src, dst, rel, operator):
    original = (src / rel).read_bytes()
    target = dst / rel
    if operator != "identity":
        image = cv2.imdecode(np.frombuffer(original, np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"expected native uint8 RGB PNG: {rel}")
        if operator == "denoise-only":
            output = cv2.fastNlMeansDenoisingColored(image, None, **OPERATORS[operator])
        else:
            blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.2)
            output = cv2.addWeighted(image, 1.15, blurred, -0.15, 0)
        if output.shape != image.shape or not cv2.imwrite(str(target), output):
            raise ValueError(f"native RGB write failed: {rel}")
    result = target.read_bytes()
    if operator == "identity" and result != original:
        raise ValueError(f"protected RGB changed during copy: {rel}")
    return {"path": rel, "source_sha256": digest(original), "output_sha256": digest(result),
            "transformed": operator != "identity", "operator": operator}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--dst", required=True, type=Path)
    parser.add_argument("--allowlist", required=True, type=Path)
    parser.add_argument("--operator", required=True, choices=OPERATORS)
    parser.add_argument("--jobs", type=int, choices=range(1, 5), default=4)
    args = parser.parse_args()
    started = time.perf_counter()
    try:
        src = args.src.resolve(strict=True)
        manifest_bytes = args.allowlist.read_bytes()
        manifest = json.loads(manifest_bytes)
        allowed, images = validate(src, args.dst, manifest)
        cv2.setNumThreads(1)
        cv2.ocl.setUseOpenCL(False)
        shutil.copytree(src, args.dst)
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            records = list(pool.map(lambda rel: process_one(
                src, args.dst, rel, args.operator if rel in allowed else "identity"), images))
        receipt = {
            "schema": "eligsir.rgb-preprocessing/v1", "status": "complete",
            "source": manifest["source"], "split": manifest["split"],
            "allowlist_sha256": digest(manifest_bytes), "opencv_version": cv2.__version__,
            "operator": {"id": args.operator, "version": 1, "parameters": OPERATORS[args.operator],
                         "color_space": "encoded sRGB, uint8 BGR in OpenCV",
                         "order": "native grid, before runtime resizing", "device": "CPU"},
            "jobs": args.jobs, "elapsed_s": time.perf_counter() - started, "images": records,
        }
        temporary = args.dst / (RECEIPT + ".tmp")
        temporary.write_text(json.dumps(receipt, indent=2) + "\n")
        temporary.replace(args.dst / RECEIPT)
    except (OSError, ValueError, cv2.error) as error:
        parser.exit(1, f"preprocessing failed: {error}\n")
    print(f"Complete native scene: {args.dst}; receipt: {RECEIPT}")


if __name__ == "__main__":
    main()
