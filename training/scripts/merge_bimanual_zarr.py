#!/usr/bin/env python3
"""Combine cropped DP3 Zarr datasets while retaining source quality labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from numcodecs import Blosc
import numpy as np
import zarr


CONTRACT_ATTRS = (
    "target_hz", "point_frame", "source_point_frame", "calibration_version",
    "state_dim", "action_dim", "state_names", "action_names", "action_semantics",
    "crop_min", "crop_max", "clip_planes", "voxel_size_m", "num_points",
)


def load_inputs(specs):
    inputs = []
    labels = set()
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Input must be QUALITY=PATH: {spec}")
        label, raw_path = spec.split("=", 1)
        if not label or "/" in label or label in labels:
            raise ValueError(f"Invalid or duplicate quality label: {label!r}")
        labels.add(label)
        path = Path(raw_path).expanduser().resolve()
        root = zarr.open_group(str(path), mode="r")
        if root.attrs.get("training_ready") is not True or root.attrs.get("conversion_complete") is not True:
            raise ValueError(f"Input is not training ready: {path}")
        points, state, action = (root[f"data/{key}"] for key in ("point_cloud", "state", "action"))
        ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
        names = root.attrs.get("source_episode_names")
        frames = len(state)
        if (points.ndim != 3 or points.shape[2] != 3 or state.ndim != 2
                or action.shape != state.shape or len(points) != frames
                or len(ends) == 0 or ends[-1] != frames
                or np.any(np.diff(np.r_[0, ends]) <= 0)
                or not isinstance(names, list) or len(names) != len(ends)):
            raise ValueError(f"Invalid episode structure: {path}")
        if any(array.dtype != np.dtype("f4") for array in (points, state, action)):
            raise ValueError(f"Input arrays must be float32: {path}")
        if points.shape[1] != root.attrs.get("num_points") or state.shape[1] != root.attrs.get("state_dim"):
            raise ValueError(f"Input shape differs from recorded attributes: {path}")
        inputs.append((label, path, root, ends, names))
    reference = inputs[0][2].attrs
    for label, path, root, _, _ in inputs[1:]:
        mismatched = [key for key in CONTRACT_ATTRS if root.attrs.get(key) != reference.get(key)]
        if mismatched:
            raise ValueError(f"{label} differs from the first input in {mismatched}: {path}")
    return inputs


def merge(output: Path, specs: list[str], exclusion_file: Path | None = None):
    inputs = load_inputs(specs)
    exclusions = {}
    if exclusion_file is not None:
        document = json.loads(exclusion_file.read_text())
        exclusions = document.get("exclude", {})
        if not isinstance(exclusions, dict) or any(not isinstance(reason, str) or not reason.strip()
                                                 for reason in exclusions.values()):
            raise ValueError("Exclusions must map QUALITY/episode to a nonempty reason")

    entries = []
    available = set()
    for label, path, root, ends, names in inputs:
        previous = 0
        for index, end in enumerate(ends):
            identity = f"{label}/{names[index]}"
            available.add(identity)
            if identity not in exclusions:
                entries.append((label, path, root, previous, int(end), identity))
            previous = int(end)
    unknown = set(exclusions) - available
    if unknown:
        raise ValueError(f"Excluded episodes were not found: {sorted(unknown)}")
    if not entries:
        raise ValueError("All episodes are excluded")

    output = output.expanduser().resolve()
    temporary = output.with_name(output.name + ".partial")
    if output.exists() or temporary.exists():
        raise FileExistsError(f"Output or partial output already exists: {output}")
    if any(output == path or output in path.parents or path in output.parents
           for _, path, _, _, _, _ in entries):
        raise ValueError("Output must not overlap an input")
    output.parent.mkdir(parents=True, exist_ok=True)
    total_frames = sum(end - start for _, _, _, start, end, _ in entries)
    point_count = inputs[0][2]["data/point_cloud"].shape[1]
    dimension = inputs[0][2]["data/state"].shape[1]
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    try:
        merged = zarr.open_group(str(temporary), mode="w")
        data, meta = merged.create_group("data"), merged.create_group("meta")
        points = data.create_dataset("point_cloud", shape=(total_frames, point_count, 3),
                                     chunks=(min(64, total_frames), point_count, 3),
                                     dtype="f4", compressor=compressor)
        state = data.create_dataset("state", shape=(total_frames, dimension),
                                    chunks=(min(1024, total_frames), dimension),
                                    dtype="f4", compressor=compressor)
        action = data.create_dataset("action", shape=(total_frames, dimension),
                                     chunks=(min(1024, total_frames), dimension),
                                     dtype="f4", compressor=compressor)
        ends_out = []
        source_names = []
        quality_names = []
        cursor = 0
        for label, _, root, start, end, identity in entries:
            for offset in range(start, end, 64):
                stop = min(offset + 64, end)
                target = slice(cursor + offset - start, cursor + stop - start)
                source = slice(offset, stop)
                points[target] = root["data/point_cloud"][source]
                state[target] = root["data/state"][source]
                action[target] = root["data/action"][source]
            cursor += end - start
            ends_out.append(cursor)
            source_names.append(identity)
            quality_names.append(label)
        meta.create_dataset("episode_ends", data=np.asarray(ends_out, dtype=np.int64),
                            compressor=compressor)
        merged.attrs.update(dict(inputs[0][2].attrs))
        merged.attrs.update(source_episode_names=source_names,
                            source_quality_names=quality_names,
                            excluded_episode_names=sorted(exclusions),
                            quality="mixed", source_datasets={label: str(path) for label, path, *_ in inputs},
                            frames=total_frames, episodes=len(entries),
                            conversion_complete=True, training_ready=True)
        temporary.rename(output)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    report = {
        "output": str(output), "frames": total_frames, "episodes": len(entries),
        "quality_counts": {label: quality_names.count(label) for label, *_ in inputs},
        "excluded": exclusions, "source_episode_names": source_names,
    }
    output.with_name(output.name + ".manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({key: report[key] for key in ("output", "frames", "episodes", "quality_counts")},
                     ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--input", action="append", required=True, metavar="QUALITY=PATH")
    parser.add_argument("--exclude-file", type=Path)
    args = parser.parse_args()
    if len(args.input) < 2:
        parser.error("Provide at least two --input QUALITY=PATH arguments")
    merge(args.output, args.input, args.exclude_file)


if __name__ == "__main__":
    main()
