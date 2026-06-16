import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import groupby

import httpx
from openai import APIConnectionError

from azure_client import get_gpt4o_model
from vision_analysis import describe_frame
_FRAME_RE = re.compile(r"frame_(\d+)")


def _frame_index(filename: str) -> int:
    match = _FRAME_RE.search(filename)
    return int(match.group(1)) if match else 0


def _load_cache(cache_path: str | None) -> dict:
    if cache_path and os.path.isfile(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_cache(cache_path: str | None, cache: dict) -> None:
    if not cache_path:
        return
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def _expand_sparse_results(sparse_results: list, all_files: list) -> list:
    """Carry each analyzed action forward to skipped frames until the next sample."""
    if not sparse_results:
        return []
    if len(sparse_results) >= len(all_files):
        return sorted(sparse_results, key=lambda r: _frame_index(r["frame"]))

    actions_by_idx = {_frame_index(r["frame"]): r["action"] for r in sparse_results}
    first_idx = min(actions_by_idx)
    last_action = actions_by_idx[first_idx]
    full = []

    for filename in all_files:
        idx = _frame_index(filename)
        if idx in actions_by_idx:
            last_action = actions_by_idx[idx]
        full.append({"frame": filename, "action": last_action})

    return full


def analyze_frames(
    frames_folder: str,
    *,
    workers: int = 6,
    analyze_every: int = 1,
    cache_path: str | None = None,
):
    """Analyze extracted frames with GPT-4o vision (parallel + optional sampling)."""
    model = get_gpt4o_model()
    all_files = sorted(f for f in os.listdir(frames_folder) if f.endswith(".jpg"))
    if not all_files:
        return []

    analyze_every = max(1, int(analyze_every))
    workers = max(1, int(workers))
    target_files = all_files[::analyze_every]

    cache = _load_cache(cache_path)
    pending = [f for f in target_files if f not in cache]
    cached_count = len(target_files) - len(pending)

    print(
        f"Mapping: Analyzing {len(target_files)}/{len(all_files)} frames "
        f"with GPT-4o ('{model}'), {workers} parallel workers..."
    )
    if analyze_every > 1:
        print(f"  Sampling every {analyze_every} extracted frame(s) to reduce API calls.")
    if cached_count:
        print(f"  Reusing {cached_count} cached frame result(s).")

    raw_results = [
        {"frame": filename, "action": cache[filename]}
        for filename in target_files
        if filename in cache
    ]

    connection_failures = 0
    failure_lock = threading.Lock()
    cache_lock = threading.Lock()
    completed = cached_count
    total_pending = len(pending)

    def analyze_one(filename: str) -> tuple[str, str]:
        nonlocal connection_failures
        file_path = os.path.join(frames_folder, filename)
        try:
            return filename, describe_frame(file_path)
        except (APIConnectionError, httpx.ConnectError, OSError) as exc:
            with failure_lock:
                connection_failures += 1
                if connection_failures >= 3:
                    raise RuntimeError(
                        "Stopping — Azure GPT-4o endpoint is unreachable (DNS/network). "
                        "Fix AZURE_OPENAI_ENDPOINT in .env and try again."
                    ) from exc
            raise

    if pending:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(analyze_one, filename): filename for filename in pending}
            for future in as_completed(futures):
                filename = futures[future]
                try:
                    _, action = future.result()
                except RuntimeError:
                    raise
                except Exception as exc:
                    print(f"  Error on {filename}: {exc}")
                    continue

                completed += 1
                raw_results.append({"frame": filename, "action": action})
                with cache_lock:
                    cache[filename] = action
                    if completed % 10 == 0 or completed == len(target_files):
                        _save_cache(cache_path, cache)
                print(f"  [{completed}/{len(target_files)}] {filename}: {action[:100]}...")

    _save_cache(cache_path, cache)
    raw_results.sort(key=lambda r: _frame_index(r["frame"]))

    if analyze_every > 1:
        raw_results = _expand_sparse_results(raw_results, all_files)

    print(f"Frame analysis complete: {len(raw_results)}/{len(all_files)} frames covered.")
    return raw_results


def consolidate_sequence(raw_results):
    print("Reducing: Consolidating sequence...")
    consolidated = []

    for action, group in groupby(raw_results, key=lambda x: x["action"]):
        group_list = list(group)
        consolidated.append(
            {
                "action": action,
                "start": group_list[0]["frame"],
                "end": group_list[-1]["frame"],
            }
        )
    return consolidated


