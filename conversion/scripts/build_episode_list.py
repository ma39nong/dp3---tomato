#!/usr/bin/env python3
"""Build a local episode list from finalized, readable clean ROS2 bags."""
import argparse
import json
from pathlib import Path
import re
import sqlite3

import yaml


def build(source_root: Path, contract_path: Path, max_episode_exclusive: int | None = None):
    contract = yaml.safe_load(contract_path.read_text())
    required = {(item["topic"], item["type"]) for item in contract["topics"].values()}
    selected, skipped = [], {}
    for episode in sorted(source_root.iterdir()):
        if not episode.is_dir():
            continue
        metadata = episode / "metadata.yaml"
        if not metadata.is_file():
            skipped[episode.name] = "missing metadata.yaml"
            continue
        number = re.fullmatch(r"episode([0-9]+)", episode.name)
        if max_episode_exclusive is not None and number and int(number.group(1)) >= max_episode_exclusive:
            skipped[episode.name] = f"episode number is not before {max_episode_exclusive}"
            continue
        try:
            state = json.loads((episode / "source_collection_state.json").read_text())
            if state.get("finalized") is not True or state.get("failures"):
                raise ValueError(f"collection validation failed: {state.get('failures', [])}")
            info = yaml.safe_load(metadata.read_text())["rosbag2_bagfile_information"]
            if info["storage_identifier"] != "sqlite3" or info.get("compression_format"):
                raise ValueError("requires an uncompressed sqlite3 bag")
            if not info.get("relative_file_paths"):
                raise ValueError("no bag database listed in metadata")
            size = 0
            for relative in info["relative_file_paths"]:
                db_path = (episode / relative).resolve()
                if not db_path.is_relative_to(episode.resolve()):
                    raise ValueError("bag database escapes episode directory")
                size += db_path.stat().st_size
                with sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True) as db:
                    found = set(db.execute("SELECT name,type FROM topics"))
                missing = required - found
                if missing:
                    raise ValueError(f"missing required topics: {sorted(missing)}")
            selected.append((size, episode.name))
        except (OSError, ValueError, KeyError, sqlite3.DatabaseError) as error:
            skipped[episode.name] = f"{type(error).__name__}: {error}"
    selected.sort(key=lambda item: (item[0], item[1]))
    return [name for _, name in selected], skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--max-episode-exclusive", type=int)
    args = parser.parse_args()
    if args.max_episode_exclusive is not None and args.max_episode_exclusive < 1:
        parser.error("--max-episode-exclusive must be positive")
    names, skipped = build(args.source_root.expanduser().resolve(), args.contract.expanduser().resolve(),
                           args.max_episode_exclusive)
    if not names:
        raise ValueError("No finalized, readable episodes found; existing list left unchanged")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text("# Finalized clean episodes, smallest first for --limit 1\n"
                         + "\n".join(names) + "\n")
    temporary.replace(args.output)
    report = args.output.with_suffix(".report.json")
    report.write_text(json.dumps({"selected": len(names), "skipped": skipped},
                                 ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"selected": len(names), "skipped": len(skipped),
                      "list": str(args.output), "report": str(report)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
