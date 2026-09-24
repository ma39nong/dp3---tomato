#!/usr/bin/env python3
"""Preview, crop and verify the packed intermediate Zarr used by this project."""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

for _thread_env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_env] = "1"

from numcodecs import Blosc
import numpy as np
from tqdm import tqdm
import zarr


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from point_sampling import (  # noqa: E402
    farthest_point_sample,
    voxel_downsample,
)
from visualize_lerobot_pointcloud import (  # noqa: E402
    crop_point_cloud,
    load_crop_config,
    normalize_planes,
    preview_subset,
    save_crop_selector_html,
)


def zarr_path(path: Path, intermediate: bool) -> Path:
    path = path.expanduser().resolve()
    if intermediate and not (path / ".zgroup").exists():
        path = path / "dataset_uncropped.zarr"
    if not (path / ".zgroup").exists():
        raise FileNotFoundError(f"Zarr group not found: {path}")
    return path


def _finite(array) -> bool:
    step = max(1, int(array.chunks[0]))
    return all(np.isfinite(array[start:start + step]).all()
               for start in range(0, len(array), step))


def validate_intermediate(
    path: Path,
    expected_dim: int = 54,
    allow_camera_frame: bool = False,
    allow_incomplete: bool = False,
):
    root = zarr.open_group(str(zarr_path(path, True)), mode="r")
    required = (
        "data/point_cloud_xyz", "data/point_cloud_offsets",
        "data/state", "data/action", "meta/episode_ends",
    )
    missing = [key for key in required if key not in root]
    if missing:
        raise ValueError(f"Intermediate Zarr is missing: {missing}")
    complete = root.attrs.get("conversion_complete") is True
    if not complete and not allow_incomplete:
        raise ValueError("Intermediate conversion_complete is not true")
    if root.attrs.get("coordinate_transform_applied") is not True:
        source_frame = root.attrs.get("source_point_frame")
        point_frame = root.attrs.get("point_frame")
        if not allow_camera_frame:
            raise ValueError(
                "Point cloud is still in the camera frame; pass --allow-camera-frame only "
                "for a fixed camera used identically during training and deployment"
            )
        if not source_frame or point_frame != source_frame:
            raise ValueError("Camera-frame data must record identical source_point_frame and point_frame")
    points = root["data/point_cloud_xyz"]
    all_offsets = root["data/point_cloud_offsets"]
    state, action = root["data/state"], root["data/action"]
    ends = root["meta/episode_ends"][:]
    if len(ends) == 0 or not np.all(np.diff(np.r_[0, ends]) > 0):
        raise ValueError("Invalid episode_ends")
    frames = int(ends[-1]) if not complete else state.shape[0]
    if state.shape[0] < frames or action.shape[0] < frames or len(all_offsets) < frames + 1:
        raise ValueError("Committed episode boundary exceeds available intermediate arrays")
    offsets = all_offsets[:frames + 1]
    if points.ndim != 2 or points.shape[1] != 3 or points.dtype != np.dtype("f4"):
        raise ValueError(f"Expected point_cloud_xyz float32 [P,3], got {points.shape}/{points.dtype}")
    if state.ndim != 2 or action.ndim != 2 or state.shape[1] != expected_dim or action.shape[1] != expected_dim:
        raise ValueError(f"Expected state/action [T,{expected_dim}], got {state.shape}/{action.shape}")
    if offsets.shape != (frames + 1,) or offsets[0] != 0 or offsets[-1] > len(points):
        raise ValueError("Invalid point_cloud_offsets")
    if complete and offsets[-1] != len(points):
        raise ValueError("Complete intermediate Zarr has unreferenced point data")
    if not np.all(np.diff(offsets) > 0):
        raise ValueError("Every intermediate frame must contain points")
    if ends[-1] != frames:
        raise ValueError("Invalid episode_ends")
    return root, offsets, ends


def representative_frames(ends: np.ndarray, frames_per_episode: int) -> list[int]:
    result, start = [], 0
    for end in ends:
        count = min(frames_per_episode, int(end) - start)
        result.extend(np.linspace(start, int(end) - 1, count, dtype=int).tolist())
        start = int(end)
    return sorted(set(result))


def episode_source_names(source: Path, count: int) -> list[str]:
    records = zarr_path(source, True).parent / "conversion/records"
    names = []
    for path in sorted(records.glob("*.json")):
        record = json.loads(path.read_text())
        name = Path(record["source"]).name
        names.extend(name if len(record["episodes"]) == 1 else f"{name}/segment{index}"
                     for index, _ in enumerate(record["episodes"]))
    return names if len(names) == count else [f"episode_index_{index}" for index in range(count)]


def _sample_points(points, crop_min, crop_max, planes, voxel_size, num_points):
    cropped = crop_point_cloud(points, crop_min, crop_max, planes)
    if len(cropped) < num_points:
        raise ValueError(
            f"Only {len(cropped)} cropped points remain, fewer than {num_points}; widen the crop"
        )
    candidates = voxel_downsample(cropped, voxel_size)
    if len(candidates) < num_points:
        candidates = cropped
    return farthest_point_sample(candidates, num_points).astype(
        np.float32, copy=False
    ), len(cropped), len(candidates)


def _parallel_init(source_path, crop_min, crop_max, planes, voxel_size, num_points):
    global _SOURCE_POINTS, _SOURCE_OFFSETS, _CROP_MIN, _CROP_MAX, _PLANES, _VOXEL_SIZE, _NUM_POINTS
    source = zarr.open_group(source_path, mode="r")
    _SOURCE_POINTS = source["data/point_cloud_xyz"]
    _SOURCE_OFFSETS = source["data/point_cloud_offsets"][:]
    _CROP_MIN, _CROP_MAX, _PLANES = crop_min, crop_max, planes
    _VOXEL_SIZE, _NUM_POINTS = voxel_size, num_points


def _parallel_block(bounds):
    start, end = bounds
    output = np.empty((end - start, _NUM_POINTS, 3), dtype=np.float32)
    crop_counts = np.empty(end - start, dtype=np.int32)
    voxel_counts = np.empty(end - start, dtype=np.int32)
    for slot, frame in enumerate(range(start, end)):
        points = _SOURCE_POINTS[_SOURCE_OFFSETS[frame]:_SOURCE_OFFSETS[frame + 1]]
        output[slot], crop_counts[slot], voxel_counts[slot] = _sample_points(
            points, _CROP_MIN, _CROP_MAX, _PLANES, _VOXEL_SIZE, _NUM_POINTS
        )
    return start, output, crop_counts, voxel_counts


def _atomic_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _source_episode_manifest(source_path: Path, ends: np.ndarray) -> list[dict]:
    path = source_path.parent / "episode_manifest.json"
    if not path.exists():
        return [{"episode_index": index, "source_bag": f"episode_index_{index}"}
                for index in range(len(ends))]
    episodes = json.loads(path.read_text())
    if len(episodes) != len(ends):
        raise ValueError("Source episode manifest does not match episode_ends")
    previous = 0
    for index, (episode, end) in enumerate(zip(episodes, ends)):
        if episode.get("episode_index") != index or episode.get("length") != int(end) - previous:
            raise ValueError(f"Source episode manifest mismatch at episode {index}")
        previous = int(end)
    return episodes


def _git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=SCRIPT_DIR.parent, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def preview(args):
    root, offsets, ends = validate_intermediate(
        args.source, args.expected_dim, args.allow_camera_frame, args.allow_incomplete
    )
    clouds, frame_ids, labels = [], [], []
    frames = representative_frames(ends, args.frames_per_episode)
    points_per_frame = min(args.points_per_frame, max(1, args.max_points // len(frames)))
    starts = np.r_[0, ends[:-1]]
    source_names = episode_source_names(args.source, len(ends))
    for slot, frame in enumerate(frames):
        points = root["data/point_cloud_xyz"][offsets[frame]:offsets[frame + 1]]
        cloud = preview_subset(points, points_per_frame, args.seed + frame)
        episode = int(np.searchsorted(ends, frame, side="right"))
        clouds.append(cloud)
        frame_ids.append(np.full(len(cloud), slot, dtype=np.int32))
        labels.append(
            f"{source_names[episode]} | episode frame {frame - starts[episode]}/"
            f"{ends[episode] - starts[episode] - 1} | global frame {frame}"
        )
    combined, combined_ids = np.concatenate(clouds), np.concatenate(frame_ids)
    if args.crop_config:
        initial_min, initial_max, initial_planes = load_crop_config(args.crop_config)
    else:
        initial_min = initial_max = None
        initial_planes = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_crop_selector_html(
        combined, args.output, initial_min, initial_max, combined_ids, labels, initial_planes,
        str((args.crop_config or Path.cwd() / "crop_config.json").expanduser().resolve()),
        camera_origin=args.camera_origin,
    )
    print(json.dumps({
        "frames_sampled": len(frames), "points_shown": len(combined),
        "xyz_min": combined.min(axis=0).tolist(), "xyz_max": combined.max(axis=0).tolist(),
        "selector": str(args.output.resolve()),
    }, ensure_ascii=False, indent=2))


def _create_output(path: Path, source, frames: int, episodes: int, dim: int, num_points: int):
    root = zarr.open_group(str(path), mode="w")
    data, meta = root.create_group("data"), root.create_group("meta")
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    pc_chunk = min(64, frames)
    vector_chunk = min(1024, frames)
    data.create_dataset("point_cloud", shape=(frames, num_points, 3),
                        chunks=(pc_chunk, num_points, 3), dtype="f4", compressor=compressor)
    data.create_dataset("state", shape=(frames, dim), chunks=(vector_chunk, dim),
                        dtype="f4", compressor=compressor)
    data.create_dataset("action", shape=(frames, dim), chunks=(vector_chunk, dim),
                        dtype="f4", compressor=compressor)
    meta.create_dataset("episode_ends", shape=(episodes,), chunks=(episodes,),
                        dtype="i8", compressor=compressor)
    root.attrs.update(dict(source.attrs))
    root.attrs.update(conversion_complete=False, training_ready=False)
    return root


def convert(args):
    source, offsets, ends = validate_intermediate(
        args.source, args.expected_dim, args.allow_camera_frame
    )
    crop_config = getattr(args, "crop_config", None)
    if crop_config:
        if args.crop_min is not None or args.crop_max is not None:
            raise ValueError("Use either --crop-config or --crop-min/--crop-max, not both")
        crop_min, crop_max, planes = load_crop_config(crop_config)
    else:
        if args.crop_min is None or args.crop_max is None:
            raise ValueError("Provide --crop-config or both --crop-min and --crop-max")
        crop_min = np.asarray(args.crop_min, dtype=np.float32)
        crop_max = np.asarray(args.crop_max, dtype=np.float32)
        planes = []
    if np.any(crop_min >= crop_max):
        raise ValueError("Every crop-min value must be smaller than crop-max")
    output = args.output.expanduser().resolve()
    source_path = zarr_path(args.source, True)
    workers = getattr(args, "workers", 1)
    chunk_frames = getattr(args, "chunk_frames", 64)
    progress_path = output.with_name(output.name + ".progress.json")
    if output == source_path or output in source_path.parents or source_path in output.parents:
        raise ValueError("Source and output Zarr paths must not overlap")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output}; pass --overwrite to replace it")
    temporary = output.with_name(output.name + ".partial")
    if temporary.exists():
        if not args.overwrite:
            raise FileExistsError(f"Previous partial output exists: {temporary}")
        shutil.rmtree(temporary)
    output.parent.mkdir(parents=True, exist_ok=True)

    frames = source["data/state"].shape[0]
    if args.expected_episodes is not None and len(ends) != args.expected_episodes:
        raise ValueError(f"Expected {args.expected_episodes} episodes, found {len(ends)}")
    if args.expected_frames is not None and frames != args.expected_frames:
        raise ValueError(f"Expected {args.expected_frames} frames, found {frames}")
    source_episodes = _source_episode_manifest(source_path, ends)
    source_names = [Path(item["source_bag"]).name for item in source_episodes]
    exclusion = None
    if args.exclusion_record:
        exclusion = json.loads(args.exclusion_record.read_text())
        if set(exclusion.get("episodes", {})).intersection(source_names):
            raise ValueError("An excluded episode is present in the source dataset")
    root = _create_output(temporary, source, frames, len(ends), args.expected_dim, args.num_points)
    crop_counts, voxel_counts = [], []
    started = time.monotonic()
    last_progress = 0.0

    def record_progress(status, processed=0, error=None):
        elapsed = time.monotonic() - started
        rate = processed / elapsed if elapsed else 0.0
        _atomic_json(progress_path, {
            "status": status, "processed_frames": processed, "total_frames": frames,
            "percent": round(processed * 100 / frames, 3), "frames_per_second": round(rate, 3),
            "eta_seconds": round((frames - processed) / rate) if rate else None,
            "workers": workers, "error": error,
        })

    record_progress("processing")
    try:
        progress = tqdm(total=frames, desc="Crop + FPS", unit="frame", mininterval=5)
        if workers == 1:
            for frame in range(frames):
                points = source["data/point_cloud_xyz"][offsets[frame]:offsets[frame + 1]]
                sampled, crop_count, voxel_count = _sample_points(
                    points, crop_min, crop_max, planes, args.voxel_size, args.num_points
                )
                root["data/point_cloud"][frame] = sampled
                crop_counts.append(crop_count)
                voxel_counts.append(voxel_count)
                progress.update(1)
                if time.monotonic() - last_progress >= 5:
                    record_progress("processing", progress.n)
                    last_progress = time.monotonic()
        else:
            blocks = [(start, min(start + chunk_frames, frames))
                      for start in range(0, frames, chunk_frames)]
            block_iter = iter(blocks)
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=workers, mp_context=context, initializer=_parallel_init,
                initargs=(str(source_path), crop_min, crop_max, planes,
                          args.voxel_size, args.num_points),
            ) as pool:
                pending = {pool.submit(_parallel_block, block) for block in
                           [next(block_iter) for _ in range(min(workers * 2, len(blocks)))]}
                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        start, sampled, crops, voxels = future.result()
                        root["data/point_cloud"][start:start + len(sampled)] = sampled
                        crop_counts.extend(crops.tolist())
                        voxel_counts.extend(voxels.tolist())
                        progress.update(len(sampled))
                        try:
                            pending.add(pool.submit(_parallel_block, next(block_iter)))
                        except StopIteration:
                            pass
                    if time.monotonic() - last_progress >= 5:
                        record_progress("processing", progress.n)
                        last_progress = time.monotonic()
        progress.close()
        root["data/state"][:] = source["data/state"][:]
        root["data/action"][:] = source["data/action"][:]
        root["meta/episode_ends"][:] = ends
        root.attrs.update(
            schema="bimanual_dp3_training_xyz/v1", source_dataset=str(zarr_path(args.source, True)),
            crop_min=crop_min.tolist(), crop_max=crop_max.tolist(), crop_applied=True,
            clip_planes=planes,
            voxel_size_m=args.voxel_size, sampling="fps", num_points=args.num_points,
            point_cloud_shape=[args.num_points, 3], state_dim=args.expected_dim,
            action_dim=args.expected_dim, conversion_complete=True, ready_for_cropping=False,
            training_ready=True, conversion_workers=workers,
            source_episode_names=source_names,
            excluded_episode_names=sorted((exclusion or {}).get("episodes", {})),
        )
        record_progress("verifying", frames)
        verify_final(temporary, args.expected_dim, args.num_points)
        step = max(1, int(root["data/state"].chunks[0]))
        for start in range(0, frames, step):
            end = min(start + step, frames)
            if not np.array_equal(root["data/state"][start:end], source["data/state"][start:end]):
                raise ValueError(f"State copy mismatch at frames {start}:{end}")
            if not np.array_equal(root["data/action"][start:end], source["data/action"][start:end]):
                raise ValueError(f"Action copy mismatch at frames {start}:{end}")
        if output.exists():
            shutil.rmtree(output)
        temporary.rename(output)
    except BaseException as exc:
        record_progress("failed", error=f"{type(exc).__name__}: {exc}")
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    report = {
        "output": str(output), "frames": frames, "episodes": len(ends),
        "crop_min": crop_min.tolist(), "crop_max": crop_max.tolist(),
        "clip_planes": planes,
        "cropped_points": {
            "min": int(np.min(crop_counts)), "median": float(np.median(crop_counts)),
            "max": int(np.max(crop_counts)),
        },
        "sampling_candidates": {
            "min": int(np.min(voxel_counts)), "median": float(np.median(voxel_counts)),
            "max": int(np.max(voxel_counts)),
        },
        "workers": workers,
    }
    _atomic_json(output.with_name(output.name + ".report.json"), report)
    previous = 0
    episode_manifest = []
    for episode, end in zip(source_episodes, ends):
        item = dict(episode)
        item.update(output_frame_start=previous, output_frame_end_exclusive=int(end))
        episode_manifest.append(item)
        previous = int(end)
    source_manifest = source_path.parent / "manifest.json"
    source_public = json.loads(source_manifest.read_text()) if source_manifest.exists() else {}
    source_bags = list(dict.fromkeys(source_names))
    if source_public.get("source_bags", source_bags) != source_bags:
        raise ValueError("Source manifest bag order does not match the episode manifest")
    final_manifest = {
        "status": "verified", "schema": "bimanual_dp3_training_xyz/v1",
        "output": str(output), "source_dataset": str(source_path),
        "source_manifest_sha256": hashlib.sha256(source_manifest.read_bytes()).hexdigest()
        if source_manifest.exists() else None,
        "crop_config": str(args.crop_config.resolve()) if args.crop_config else None,
        "crop_config_sha256": hashlib.sha256(args.crop_config.read_bytes()).hexdigest()
        if args.crop_config else None,
        "converter_git_commit": _git_revision(), "frames": frames, "episodes": len(ends),
        "parameters": {"workers": workers, "chunk_frames": chunk_frames,
                       "voxel_size_m": args.voxel_size, "num_points": args.num_points},
        "excluded": exclusion, "source_checksums": source_public.get("source_checksums", {}),
        "episode_manifest": episode_manifest,
    }
    _atomic_json(output.with_name(output.name + ".manifest.json"), final_manifest)
    included_path = output.with_name(output.name + ".included_episodes.txt")
    included_path.write_text("\n".join(source_names) + "\n", encoding="utf-8")
    record_progress("passed", frames)


def verify_final(path: Path, expected_dim: int, num_points: int):
    root = zarr.open_group(str(zarr_path(path, False)), mode="r")
    point_cloud = root["data/point_cloud"]
    state, action = root["data/state"], root["data/action"]
    ends = root["meta/episode_ends"][:]
    frames = len(point_cloud)
    expected = (frames, num_points, 3)
    if point_cloud.shape != expected or point_cloud.dtype != np.dtype("f4"):
        raise ValueError(f"Expected point_cloud {expected}/float32, got {point_cloud.shape}/{point_cloud.dtype}")
    if state.shape != (frames, expected_dim) or action.shape != (frames, expected_dim):
        raise ValueError(f"Expected state/action [T,{expected_dim}], got {state.shape}/{action.shape}")
    if len(ends) == 0 or ends[-1] != frames or not np.all(np.diff(np.r_[0, ends]) > 0):
        raise ValueError("Invalid final episode_ends")
    if not all(_finite(array) for array in (point_cloud, state, action)):
        raise ValueError("Final Zarr contains NaN or infinity")
    crop_min = np.asarray(root.attrs["crop_min"], dtype=np.float32)
    crop_max = np.asarray(root.attrs["crop_max"], dtype=np.float32)
    planes = normalize_planes(root.attrs.get("clip_planes", []))
    step = max(1, int(point_cloud.chunks[0]))
    for start in range(0, frames, step):
        points = point_cloud[start:start + step]
        if np.any(points < crop_min - 1e-6) or np.any(points > crop_max + 1e-6):
            raise ValueError("Final point cloud contains points outside its recorded AABB")
        xyz = points.reshape(-1, 3)
        for plane in planes:
            if np.any(xyz @ plane["normal"] + plane["offset"] < plane["margin"] - 1e-6):
                raise ValueError(f"Final point cloud violates clip plane {plane['name']}")
    if root.attrs.get("training_ready") is not True:
        raise ValueError("training_ready is not true")
    result = {"status": "passed", "frames": frames, "episodes": len(ends),
              "point_cloud": list(point_cloud.shape), "state": list(state.shape),
              "action": list(action.shape)}
    print(json.dumps(result, ensure_ascii=False))
    return result


def self_check():
    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory)
        source_path = base / "delivery/dataset_uncropped.zarr"
        source = zarr.open_group(str(source_path), mode="w")
        data, meta = source.create_group("data"), source.create_group("meta")
        rng = np.random.default_rng(7)
        clouds = rng.uniform(-0.5, 0.5, size=(2400, 3)).astype(np.float32)
        data.create_dataset("point_cloud_xyz", data=clouds, chunks=(1200, 3))
        data.create_dataset("point_cloud_offsets", data=np.array([0, 1200, 2400], dtype=np.int64))
        data.create_dataset("state", data=np.zeros((2, 54), dtype=np.float32))
        data.create_dataset("action", data=np.ones((2, 54), dtype=np.float32))
        meta.create_dataset("episode_ends", data=np.array([2], dtype=np.int64))
        source.attrs.update(conversion_complete=True, coordinate_transform_applied=True)
        plane = {"name": "keep_positive_x", "normal": [1, 0, 0], "offset": 0, "margin": 0.1}
        crop_config = base / "crop_config.json"
        crop_config.write_text(json.dumps({
            "crop_min": [-1, -1, -1], "crop_max": [1, 1, 1], "planes": [plane],
        }))
        output = base / "training.zarr"
        arguments = dict(
            source=base / "delivery", output=output,
            crop_min=None, crop_max=None, voxel_size=0.01,
            crop_config=crop_config, num_points=64, expected_dim=54, overwrite=False,
            allow_camera_frame=False, workers=1, chunk_frames=2,
            expected_episodes=1, expected_frames=2, exclusion_record=None,
        )
        convert(argparse.Namespace(**arguments))
        result = verify_final(output, 54, 64)
        assert result["point_cloud"] == [2, 64, 3]
        assert np.all(zarr.open_group(str(output), mode="r")["data/point_cloud"][:, :, 0] >= 0.1)
        parallel_output = base / "training_parallel.zarr"
        arguments.update(output=parallel_output, workers=2, chunk_frames=1)
        convert(argparse.Namespace(**arguments))
        single = zarr.open_group(str(output), mode="r")
        parallel = zarr.open_group(str(parallel_output), mode="r")
        for key in ("data/point_cloud", "data/state", "data/action", "meta/episode_ends"):
            assert np.array_equal(single[key][:], parallel[key][:]), key
        selector = base / "selector.html"
        save_crop_selector_html(
            clouds[:4], selector, frame_ids=np.array([0, 0, 1, 1]),
            frame_labels=["episode1 frame0", "episode1 frame1"],
        )
        html = selector.read_text()
        assert 'id="frameSelect"' in html and "episode1 frame1" in html
        selected = crop_point_cloud(clouds, np.full(3, -1), np.ones(3), [plane])
        assert len(selected) and np.all(selected[:, 0] >= 0.1)
        save_crop_selector_html(clouds[:4], selector, initial_planes=[plane])
        assert '"initialPlanes":[{"name":"keep_positive_x"' in selector.read_text()
    print("self-check passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    show = sub.add_parser("preview", help="Create an interactive AABB/plane crop editor")
    show.add_argument("source", type=Path)
    show.add_argument("--output", type=Path, default=Path("pointcloud_preview/bimanual_crop_selector.html"))
    show.add_argument("--frames-per-episode", type=int, default=3)
    show.add_argument("--points-per-frame", type=int, default=1000)
    show.add_argument("--max-points", type=int, default=100000)
    show.add_argument("--seed", type=int, default=0)
    show.add_argument("--expected-dim", type=int, default=54)
    show.add_argument("--allow-camera-frame", action="store_true")
    show.add_argument(
        "--allow-incomplete", action="store_true",
        help="Preview only fully committed episodes while conversion is still running",
    )
    show.add_argument("--crop-config", type=Path, help="Reload a previously exported crop_config.json")
    show.add_argument("--camera-origin", action="store_true", help="Show a camera marker at (0,0,0) facing +Z")
    build = sub.add_parser("convert", help="Crop and sample an intermediate Zarr into training format")
    build.add_argument("source", type=Path)
    build.add_argument("output", type=Path)
    build.add_argument("--crop-min", nargs=3, type=float)
    build.add_argument("--crop-max", nargs=3, type=float)
    build.add_argument("--crop-config", type=Path, help="AABB and clipping planes exported by preview")
    build.add_argument("--voxel-size", type=float, default=0.005)
    build.add_argument("--num-points", type=int, default=1024)
    build.add_argument("--expected-dim", type=int, default=54)
    build.add_argument("--allow-camera-frame", action="store_true")
    build.add_argument("--overwrite", action="store_true")
    build.add_argument("--workers", type=int, default=1)
    build.add_argument("--chunk-frames", type=int, default=64)
    build.add_argument("--expected-episodes", type=int)
    build.add_argument("--expected-frames", type=int)
    build.add_argument("--exclusion-record", type=Path)
    check = sub.add_parser("verify", help="Validate a final fixed-size training Zarr")
    check.add_argument("path", type=Path)
    check.add_argument("--num-points", type=int, default=1024)
    check.add_argument("--expected-dim", type=int, default=54)
    sub.add_parser("self-check", help="Run a synthetic end-to-end conversion check")
    args = parser.parse_args()
    if getattr(args, "num_points", 1) < 1 or getattr(args, "expected_dim", 1) < 1:
        parser.error("dimensions must be positive")
    for name in ("frames_per_episode", "points_per_frame", "max_points"):
        if getattr(args, name, 1) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if getattr(args, "voxel_size", 1.0) <= 0:
        parser.error("--voxel-size must be positive")
    if getattr(args, "workers", 1) < 1 or getattr(args, "chunk_frames", 1) < 1:
        parser.error("--workers and --chunk-frames must be positive")
    if args.command == "convert" and (args.crop_min is None) != (args.crop_max is None):
        parser.error("--crop-min and --crop-max must be provided together")
    if args.command == "self-check":
        self_check()
    elif args.command == "preview":
        preview(args)
    elif args.command == "convert":
        convert(args)
    else:
        verify_final(args.path, args.expected_dim, args.num_points)


if __name__ == "__main__":
    main()
