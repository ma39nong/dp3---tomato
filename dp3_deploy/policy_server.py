#!/usr/bin/env python3
"""Serve Epoch20 EMA predictions over a local Unix socket; no robot transport."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket

from inference import Predictor
from policy_socket import REQUEST_BYTES, MAX_RESPONSE_BYTES, decode_request, recv_message, send_message


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    path = args.socket.expanduser().resolve()
    if path.exists():
        raise FileExistsError(f"Refusing to replace existing socket path: {path}")
    if not path.parent.is_dir():
        raise FileNotFoundError(f"Create a private socket directory first: {path.parent}")
    predictor = Predictor(args.device)
    print(f"Synthetic policy warmup: {predictor.warmup():.1f} ms", flush=True)
    old_umask = os.umask(0o177)
    try:
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(path))
    finally:
        os.umask(old_umask)
    server.listen(1)
    print(f"Prediction-only policy socket ready: {path}", flush=True)
    try:
        while True:
            connection, _ = server.accept()
            with connection:
                while True:
                    try:
                        payload = recv_message(connection, REQUEST_BYTES)
                    except (EOFError, ConnectionError, ValueError, OSError):
                        break
                    try:
                        cloud, state = decode_request(payload)
                        result = predictor.predict(cloud, state)
                        response = {"action_rad": result["action"].tolist(),
                                    "inference_latency_ms": result["latency_ms"],
                                    "checkpoint_epoch": predictor.cfg["checkpoint_epoch"],
                                    "ema_weights_sha256": predictor.cfg["ema_weights_sha256"],
                                    "semantics": result["semantics"],
                                    "sent_robot_command": False}
                    except Exception as exc:
                        response = {"error": str(exc), "sent_robot_command": False}
                    try:
                        send_message(connection, json.dumps(response, allow_nan=False).encode(),
                                     MAX_RESPONSE_BYTES)
                    except (ConnectionError, OSError):
                        break
    finally:
        server.close()
        path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
