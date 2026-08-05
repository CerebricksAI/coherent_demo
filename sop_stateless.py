import json
import os
import queue
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import groupby

import cv2
import httpx
from openai import APIConnectionError

from azure_client import get_gpt4o_model
from vision_analysis import describe_frame
_FRAME_RE = re.compile(r"frame_(\d+)")


def _frame_index(filename: str) -> int:
    match = _FRAME_RE.search(filename)
    return int(match.group(1)) if match else 0


def _frames_similar(path_a: str, path_b: str, threshold: float = 0.97) -> bool:
    """Fast visual similarity check to skip near-duplicate frames before GPT calls."""
    img_a = cv2.imread(path_a, cv2.IMREAD_GRAYSCALE)
    img_b = cv2.imread(path_b, cv2.IMREAD_GRAYSCALE)
    if img_a is None or img_b is None:
        return False
    size = (64, 64)
    a = cv2.resize(img_a, size)
    b = cv2.resize(img_b, size)
    diff = cv2.absdiff(a, b).mean() / 255.0
    return (1.0 - diff) >= threshold


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


def analyze_frames_from_queue(
    frames_folder: str,
    frame_queue: queue.Queue,
    *,
    workers: int = 6,
    analyze_every: int = 1,
    cache_path: str | None = None,
):
    """
    Analyze frames as they arrive on frame_queue (None sentinel ends stream).
    Overlaps with frame extraction for lower latency.
    """
    model = get_gpt4o_model()
    analyze_every = max(1, int(analyze_every))
    workers = max(1, int(workers))

    cache = _load_cache(cache_path)
    all_files: list[str] = []
    raw_results: list[dict] = []
    pending_futures: dict = {}
    connection_failures = 0
    failure_lock = threading.Lock()
    cache_lock = threading.Lock()
    results_lock = threading.Lock()
    last_analyzed_path: str | None = None
    last_action = "No action."
    skipped_similar = 0
    completed = 0

    print(
        f"Mapping: Streaming vision analysis with GPT-4o ('{model}'), "
        f"{workers} parallel workers..."
    )
    if analyze_every > 1:
        print(f"  Sampling every {analyze_every} extracted frame(s).")

    def analyze_one(filename: str, file_path: str) -> tuple[str, str]:
        nonlocal connection_failures
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

    with ThreadPoolExecutor(max_workers=workers) as executor:
        while True:
            while pending_futures:
                done = [f for f in pending_futures if f.done()]
                if not done:
                    break
                for future in done:
                    filename = pending_futures.pop(future)
                    try:
                        _, action = future.result()
                    except RuntimeError:
                        raise
                    except Exception as exc:
                        print(f"  Error on {filename}: {exc}")
                        continue

                    completed += 1
                    with results_lock:
                        raw_results.append({"frame": filename, "action": action})
                    with cache_lock:
                        cache[filename] = action
                        if completed % 10 == 0:
                            _save_cache(cache_path, cache)
                    print(f"  [{completed}] {filename}: {action[:100]}...")

            try:
                item = frame_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            if item is None:
                break

            all_files.append(item)
            idx = len(all_files) - 1
            if idx % analyze_every != 0:
                continue

            file_path = os.path.join(frames_folder, item)
            if item in cache:
                action = cache[item]
                with results_lock:
                    raw_results.append({"frame": item, "action": action})
                last_analyzed_path = file_path
                last_action = action
                completed += 1
                continue

            if last_analyzed_path and _frames_similar(last_analyzed_path, file_path):
                skipped_similar += 1
                with results_lock:
                    raw_results.append({"frame": item, "action": last_action})
                with cache_lock:
                    cache[item] = last_action
                last_analyzed_path = file_path
                completed += 1
                continue

            last_analyzed_path = file_path
            pending_futures[executor.submit(analyze_one, item, file_path)] = item

        for future in as_completed(list(pending_futures)):
            filename = pending_futures[future]
            try:
                _, action = future.result()
            except RuntimeError:
                raise
            except Exception as exc:
                print(f"  Error on {filename}: {exc}")
                continue

            completed += 1
            with results_lock:
                raw_results.append({"frame": filename, "action": action})
            with cache_lock:
                cache[filename] = action
            print(f"  [{completed}] {filename}: {action[:100]}...")

    _save_cache(cache_path, cache)
    raw_results.sort(key=lambda r: _frame_index(r["frame"]))

    if analyze_every > 1:
        raw_results = _expand_sparse_results(raw_results, all_files)
    elif len(raw_results) < len(all_files):
        # Fill gaps for frames skipped by similarity with last known action
        actions_by_idx = {_frame_index(r["frame"]): r["action"] for r in raw_results}
        if actions_by_idx:
            first_idx = min(actions_by_idx)
            carry = actions_by_idx[first_idx]
            full = []
            for filename in all_files:
                idx = _frame_index(filename)
                if idx in actions_by_idx:
                    carry = actions_by_idx[idx]
                full.append({"frame": filename, "action": carry})
            raw_results = full

    if skipped_similar:
        print(f"  Skipped {skipped_similar} near-duplicate frame(s) (visual similarity).")
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


