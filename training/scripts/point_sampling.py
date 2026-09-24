"""Small deterministic point-cloud sampling helpers."""
import numpy as np


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if voxel_size <= 0 or len(points) == 0:
        return points
    coordinates = np.floor(points / voxel_size).astype(np.int32)
    _, first = np.unique(coordinates, axis=0, return_index=True)
    return points[np.sort(first)]


def farthest_point_sample(points: np.ndarray, num_points: int) -> np.ndarray:
    if len(points) == 0:
        raise ValueError("Cannot sample an empty point cloud")
    if len(points) <= num_points:
        return points[np.resize(np.arange(len(points)), num_points)]
    selected = np.empty(num_points, dtype=np.int64)
    min_distance = np.full(len(points), np.inf, dtype=np.float32)
    farthest = int(np.argmax(np.sum((points - points.mean(axis=0)) ** 2, axis=1)))
    for index in range(num_points):
        selected[index] = farthest
        delta = points - points[farthest]
        np.minimum(min_distance, np.einsum("ij,ij->i", delta, delta), out=min_distance)
        farthest = int(np.argmax(min_distance))
    return points[selected]
