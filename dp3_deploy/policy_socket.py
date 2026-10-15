"""Local, prediction-only Unix socket protocol shared by ROS and model processes."""
from __future__ import annotations

import json
import socket
import struct
import time

import numpy as np


REQUEST_BYTES = (2 * 4096 * 3 + 2 * 54) * 4
MAX_RESPONSE_BYTES = 100_000


def _exact(sock: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        data = sock.recv(size - len(chunks))
        if not data:
            raise EOFError("Policy socket closed before message completed")
        chunks.extend(data)
    return bytes(chunks)


def request_bytes(point_cloud: np.ndarray, state: np.ndarray) -> bytes:
    cloud = np.asarray(point_cloud, dtype="<f4")
    joints = np.asarray(state, dtype="<f4")
    if cloud.shape != (2, 4096, 3) or joints.shape != (2, 54):
        raise ValueError("Expected [2,4096,3] point cloud and [2,54] state")
    if not np.isfinite(cloud).all() or not np.isfinite(joints).all():
        raise ValueError("Nonfinite observation")
    return cloud.tobytes(order="C") + joints.tobytes(order="C")


def decode_request(payload: bytes) -> tuple[np.ndarray, np.ndarray]:
    if len(payload) != REQUEST_BYTES:
        raise ValueError(f"Invalid policy request size: {len(payload)}")
    values = np.frombuffer(payload, dtype="<f4")
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite policy request")
    split = 2 * 4096 * 3
    return values[:split].reshape(2, 4096, 3).copy(), values[split:].reshape(2, 54).copy()


def send_message(sock: socket.socket, payload: bytes, limit: int) -> None:
    if not 0 < len(payload) <= limit:
        raise ValueError("Policy message exceeds limit")
    sock.sendall(struct.pack("!I", len(payload)) + payload)


def recv_message(sock: socket.socket, limit: int) -> bytes:
    length = struct.unpack("!I", _exact(sock, 4))[0]
    if not 0 < length <= limit:
        raise ValueError("Invalid policy message length")
    return _exact(sock, length)


class SocketPredictor:
    """Minimal ROS-side client; imports NumPy but no Torch or model code."""

    def __init__(self, path: str, expected_sha256: str, timeout_s: float = 2.0):
        self.expected_sha256 = expected_sha256
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket.settimeout(timeout_s)
        self.socket.connect(path)

    def close(self) -> None:
        self.socket.close()

    def predict(self, point_cloud: np.ndarray, state: np.ndarray) -> dict:
        payload = request_bytes(point_cloud, state)
        start = time.perf_counter()
        send_message(self.socket, payload, REQUEST_BYTES)
        response = json.loads(recv_message(self.socket, MAX_RESPONSE_BYTES))
        if response.get("error"):
            raise RuntimeError(response["error"])
        if (response.get("ema_weights_sha256") != self.expected_sha256 or
                response.get("checkpoint_epoch") != 20 or
                response.get("semantics") != "absolute_joint_target_rad"):
            raise ValueError("Policy server checkpoint or output contract mismatch")
        action = np.asarray(response["action_rad"], dtype=np.float32)
        if action.shape != (4, 54) or not np.isfinite(action).all():
            raise ValueError("Invalid policy server response")
        return {"action": action, "latency_ms": float(response["inference_latency_ms"]),
                "roundtrip_latency_ms": (time.perf_counter() - start) * 1000,
                "semantics": "absolute_joint_target_rad", "sent_robot_command": False}
