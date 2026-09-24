"""User-maintained episode lists, separate from immutable conversion settings."""
from collections import defaultdict
from pathlib import Path
import hashlib
import re


def read_episode_list(path):
    path = Path(path).expanduser().resolve()
    payload = path.read_bytes()
    names, seen = [], {}
    for line_number, line in enumerate(payload.decode("utf-8-sig").splitlines(), 1):
        content = line.split("#", 1)[0].strip()
        for token in re.split(r"[\s,，]+", content):
            if not token:
                continue
            numeric = re.fullmatch(r"(?:episode)?([0-9]+)(?:-(?:episode)?([0-9]+))?", token)
            if numeric:
                start = int(numeric.group(1))
                end = int(numeric.group(2)) if numeric.group(2) is not None else start
                if end < start:
                    raise ValueError(f"{path.name}:{line_number}: reversed range {token!r}; start must be <= end")
                expanded = [f"episode{number}" for number in range(start, end + 1)]
            elif re.fullmatch(r"[0-9-]+", token):
                raise ValueError(
                    f"{path.name}:{line_number}: invalid numeric range {token!r}; "
                    "use 18, episode18 or 34-39"
                )
            elif re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", token):
                expanded = [token]
            else:
                raise ValueError(
                    f"{path.name}:{line_number}: invalid bag name {token!r}; "
                    "use a directory basename, 18, episode18 or 34-39"
                )
            for name in expanded:
                if name in seen:
                    raise ValueError(f"{path.name}:{line_number}: duplicate {name} (first seen on line {seen[name]})")
                names.append(name)
                seen[name] = line_number
    if not names:
        raise ValueError(f"Episode list is empty: {path}")
    return {"list_path": str(path), "list_sha256": hashlib.sha256(payload).hexdigest(),
            "requested_episodes": names}


def select_episodes(candidates, list_path, skip_ineligible=False):
    selection = read_episode_list(list_path)
    by_name = defaultdict(list)
    for candidate in candidates:
        by_name[Path(candidate["path"]).name].append(candidate)
    selected, skipped, errors = [], [], []
    for name in selection["requested_episodes"]:
        matches = by_name[name]
        if not matches:
            errors.append(f"{name}: no bag with metadata.yaml found under source_root")
        elif len(matches) != 1:
            errors.append(f"{name}: ambiguous name; narrow --source-root ({[m['path'] for m in matches]})")
        elif not matches[0]["eligible"]:
            skipped.append({"episode": name, "source": matches[0]["path"], "reason": matches[0]["reason"]})
        else:
            selected.append(matches[0])
    if errors:
        raise ValueError("Invalid episode list:\n" + "\n".join(errors))
    if skipped and not skip_ineligible:
        details = "\n".join(f"{s['episode']}: {s['reason']}" for s in skipped)
        raise ValueError("Listed bags failed collection validation:\n" + details
                         + "\nFix the list, or explicitly use --skip-ineligible to omit these bags. Freshness rules stay unchanged.")
    selection.update(selected_episodes=[Path(item["path"]).name for item in selected],
                     skipped=skipped, skip_ineligible=skip_ineligible)
    return selected, selection
