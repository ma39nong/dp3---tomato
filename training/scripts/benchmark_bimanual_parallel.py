#!/usr/bin/env python3
"""Benchmark parallel crop + voxel + FPS on representative intermediate frames."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import zarr

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from point_sampling import farthest_point_sample, voxel_downsample  # noqa: E402
from visualize_lerobot_pointcloud import crop_point_cloud, load_crop_config  # noqa: E402


def _init_worker(source: str, crop_config: str, voxel_size: float, num_points: int):
    global POINTS, OFFSETS, CROP_MIN, CROP_MAX, PLANES, VOXEL_SIZE, NUM_POINTS
    root = zarr.open_group(source, mode="r")
    POINTS = root["data/point_cloud_xyz"]
    OFFSETS = root["data/point_cloud_offsets"][:]
    CROP_MIN, CROP_MAX, PLANES = load_crop_config(Path(crop_config))
    VOXEL_SIZE, NUM_POINTS = voxel_size, num_points


def _process_frames(frame_ids: list[int]) -> dict:
    digest = 0.0
    for frame in frame_ids:
        points = POINTS[int(OFFSETS[frame]):int(OFFSETS[frame + 1])]
        cropped = crop_point_cloud(points, CROP_MIN, CROP_MAX, PLANES)
        if len(cropped) < NUM_POINTS:
            raise ValueError(f"Frame {frame} has only {len(cropped)} cropped points")
        candidates = voxel_downsample(cropped, VOXEL_SIZE)
        if len(candidates) < NUM_POINTS:
            candidates = cropped
        sampled = farthest_point_sample(candidates, NUM_POINTS)
        digest += float(sampled[0].sum())
    return {"frames": len(frame_ids), "digest": digest}


def _representative_frames(ends: np.ndarray, target: int) -> list[int]:
    starts = np.r_[0, ends[:-1]]
    base, extra = divmod(target, len(ends))
    frames = []
    for index, (start, end) in enumerate(zip(starts, ends)):
        count = min(int(end - start), base + (index < extra))
        frames.extend(np.linspace(start, end - 1, count, dtype=int).tolist())
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--crop-config", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=1024)
    parser.add_argument("--workers", nargs="+", type=int, default=[8, 12, 16])
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--voxel-size", type=float, default=0.005)
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    if not (source / ".zgroup").exists():
        source /= "dataset_uncropped.zarr"
    root = zarr.open_group(str(source), mode="r")
    ends = root["meta/episode_ends"][:]
    frame_ids = _representative_frames(ends, args.frames)
    order = args.workers + list(reversed(args.workers))
    order = (order * ((args.rounds * len(args.workers) + len(order) - 1) // len(order)))[
        :args.rounds * len(args.workers)
    ]
    print(json.dumps({"status": "started", "frames": len(frame_ids), "episodes": len(ends),
                      "order": order}), flush=True)

    runs = []
    context = multiprocessing.get_context("spawn")
    for run_index, workers in enumerate(order, 1):
        chunks = [part.tolist() for part in np.array_split(frame_ids, workers) if len(part)]
        started = time.perf_counter()
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=context,
            initializer=_init_worker,
            initargs=(str(source), str(args.crop_config.resolve()), args.voxel_size, args.num_points),
        ) as pool:
            results = list(pool.map(_process_frames, chunks))
        elapsed = time.perf_counter() - started
        fps = len(frame_ids) / elapsed
        run = {"run": run_index, "workers": workers, "seconds": elapsed,
               "frames_per_second": fps,
               "projected_hours": int(root["data/state"].shape[0]) / fps / 3600,
               "digest": sum(item["digest"] for item in results)}
        runs.append(run)
        print(json.dumps(run), flush=True)

    summary = []
    for workers in args.workers:
        selected = [run for run in runs if run["workers"] == workers]
        summary.append({
            "workers": workers,
            "median_frames_per_second": float(np.median([run["frames_per_second"] for run in selected])),
            "median_projected_hours": float(np.median([run["projected_hours"] for run in selected])),
        })
    result = {"status": "passed", "frames": len(frame_ids), "runs": runs,
              "summary": summary, "best": min(summary, key=lambda item: item["median_projected_hours"])}
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text, flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
