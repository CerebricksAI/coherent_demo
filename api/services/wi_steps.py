import json
import os
import re
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher

from api.services.step_enricher import extract_instruments
from api.services.video_clips import cut_video_clip
from azure_client import get_client, get_writer_model
from report_builder import (
    enrich_visual_sequence,
    parse_work_instructions_blocks,
    timestamp_label_to_seconds,
)
from time_utils import seconds_to_timestamp

WI_PLACEHOLDER_TEXT = "View this segment in the video."
WI_TIMESTAMPED_STEP_RE = re.compile(
    r"(?:^|\n)"
    r"\s*(?:\d+\.\s*)?"
    r"(?:\*\*)?"
    r"\[(?P<start>[^\]]+?)\s*-\s*(?P<end>[^\]]+?)\]"
    r"(?:\*\*)?"
    r"[ \t]*"
    r"(?P<body>.*?)"
    r"(?=(?:\n\s*(?:\d+\.\s*)?(?:\*\*)?\[)|\Z)",
    re.DOTALL,
)

MIN_STEP_SECONDS = 8.0
MAX_STEP_SECONDS = 32.0
MIN_SPEECH_SPLIT_SECONDS = 4.0
WI_SIMILARITY_THRESHOLD = 0.68
MIN_SPEECH_CHARS = 12

ENUMERATION_RE = re.compile(
    r"\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|next)\b",
    re.IGNORECASE,
)
ENUMERATION_SPLIT_RE = re.compile(
    r"(?=\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|next)\b)",
    re.IGNORECASE,
)
ORDINALS_RE = re.compile(
    r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b",
    re.IGNORECASE,
)

WEAK_WI_EXACT = frozenset(
    {
        "complete the task shown in this video segment.",
        "view this segment in the video.",
        WI_PLACEHOLDER_TEXT.lower(),
        "no action.",
    }
)

META_WI_PATTERNS = (
    re.compile(r"^this clip (demonstrates|shows)\b", re.I),
    re.compile(r"^in this (clip|segment)\b", re.I),
    re.compile(r"^follow the operator narration\b", re.I),
    re.compile(r"^during this segment\b", re.I),
)
TIMESTAMP_SUBITEM_RE = re.compile(r"^\s*at\s+\d+:\d+", re.I)
INLINE_TIMESTAMP_RE = re.compile(
    r"\*{0,2}\[?\d+:\d{2}(?:\.\d{2})?\s*-\s*\d+:\d{2}(?:\.\d{2})?\]?\*{0,2}",
    re.IGNORECASE,
)


def _actions_similar(a: str, b: str) -> bool:
    a_norm = re.sub(r"\s+", " ", (a or "").lower()).strip()
    b_norm = re.sub(r"\s+", " ", (b or "").lower()).strip()
    if not a_norm or not b_norm:
        return True
    return SequenceMatcher(None, a_norm, b_norm).ratio() >= 0.45


def _dedupe_actions(actions: list[str]) -> list[str]:
    unique: list[str] = []
    for action in actions:
        action = (action or "").strip()
        if not action:
            continue
        if any(_actions_similar(action, existing) for existing in unique):
            continue
        unique.append(action)
    return unique


def _usable_observations(observations: list[str]) -> list[str]:
    """Drop empty/no-worker frame notes; keep actionable descriptions."""
    kept = []
    for obs in observations:
        obs = (obs or "").strip()
        if not obs:
            continue
        lower = obs.lower()
        if lower in {"no action", "no action."}:
            continue
        if "not visible" in lower and len(obs) < 80:
            continue
        kept.append(obs)
    return kept


def _normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", (text or "").lower()).strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def _is_idle_action(text: str) -> bool:
    """True when a frame note carries no actionable work."""
    text = (text or "").strip().lower()
    if not text:
        return True
    idle_phrases = (
        "no action",
        "not visible",
        "worker is not visible",
        "nothing relevant",
        "no relevant",
        "static frame",
        "empty frame",
        "idle",
    )
    return any(p in text for p in idle_phrases)


def _is_important_segment(segment: dict) -> bool:
    """Keep segments with actionable visual content and/or meaningful speech."""
    observations = segment.get("visual_observations") or []
    speech = (segment.get("spoken_narration") or "").strip()
    raw_actions = segment.get("raw_actions") or []

    if observations:
        return True
    if len(speech) >= MIN_SPEECH_CHARS:
        return True
    # All raw actions are idle/noise — skip even if speech is a mumbled fragment
    if raw_actions and all(_is_idle_action(a) for a in raw_actions):
        return False
    return False


def _overlaps_time(start_a: float, end_a: float, start_b: float, end_b: float) -> bool:
    return start_a < end_b and end_a > start_b


def _has_enumeration(text: str) -> bool:
    return len(ENUMERATION_RE.findall(text or "")) >= 2


def _segment_similar(a: dict, b: dict) -> bool:
    """True when two adjacent segments describe essentially the same work."""
    a_speech = a.get("spoken_narration") or ""
    b_speech = b.get("spoken_narration") or ""
    if _has_enumeration(a_speech) or _has_enumeration(b_speech):
        return False
    if len(a.get("timed_speech") or []) > 1 or len(b.get("timed_speech") or []) > 1:
        return False

    a_obs = " ".join(a.get("visual_observations") or [])
    b_obs = " ".join(b.get("visual_observations") or [])

    if a_obs and b_obs and _actions_similar(a_obs, b_obs):
        return True
    if a_speech and b_speech:
        if _actions_similar(a_speech, b_speech):
            return True
    if a_obs and b_obs and a_speech and b_speech:
        combined_a = f"{a_obs} {a_speech}"
        combined_b = f"{b_obs} {b_speech}"
        if SequenceMatcher(None, _normalize_text(combined_a), _normalize_text(combined_b)).ratio() >= WI_SIMILARITY_THRESHOLD:
            return True
    return False


def wi_text_similar(a: str, b: str) -> bool:
    if not a or not b:
        return False
    return SequenceMatcher(None, _normalize_text(a), _normalize_text(b)).ratio() >= WI_SIMILARITY_THRESHOLD


def dedupe_similar_steps(steps: list[dict]) -> list[dict]:
    """Drop steps whose WI text closely repeats a prior step."""
    ordered = sorted(steps, key=lambda s: float(s.get("start_sec", 0)))
    kept: list[dict] = []
    last_wi = ""
    for step in ordered:
        wi = (step.get("text") or "").strip()
        if not wi:
            continue
        if kept and wi_text_similar(wi, last_wi):
            print(
                f"  Dropping duplicate WI at {step.get('start_label', '?')} "
                f"(similar to previous step)."
            )
            continue
        kept.append(step)
        last_wi = wi
    return kept


def format_steps_as_work_instructions(steps: list[dict]) -> str:
    """Build a plain-text WI document from segment-aligned steps."""
    lines: list[str] = []
    for index, step in enumerate(steps, start=1):
        text = (step.get("text") or "").strip()
        if not text:
            continue
        start = step.get("start_label", "")
        end = step.get("end_label", "")
        lines.append(f"{index}. [{start} - {end}] {text}")
        instruments = step.get("instruments") or []
        if instruments:
            lines.append(f"   Instruments used: {', '.join(instruments)}")
    return "\n\n".join(lines)


def merge_adjacent_similar_segments(segments: list[dict]) -> list[dict]:
    """Merge consecutive segments that describe the same action/speech."""
    if not segments:
        return []

    merged: list[dict] = [dict(segments[0])]
    for seg in segments[1:]:
        prev = merged[-1]
        if _segment_similar(prev, seg):
            prev["end_sec"] = seg["end_sec"]
            prev["end_label"] = seg["end_label"]
            prev_obs = list(prev.get("visual_observations") or [])
            for obs in seg.get("visual_observations") or []:
                if obs not in prev_obs:
                    prev_obs.append(obs)
            prev["visual_observations"] = _dedupe_actions(prev_obs)
            prev_speech = prev.get("spoken_narration") or ""
            new_speech = seg.get("spoken_narration") or ""
            if new_speech and new_speech not in prev_speech:
                prev["spoken_narration"] = f"{prev_speech} {new_speech}".strip()
            prev_ts = list(prev.get("timed_speech") or [])
            prev_ts.extend(seg.get("timed_speech") or [])
            prev["timed_speech"] = prev_ts
            prev_tv = list(prev.get("timed_visual") or [])
            for tv in seg.get("timed_visual") or []:
                if tv not in prev_tv:
                    prev_tv.append(tv)
            prev["timed_visual"] = prev_tv
            raw = list(prev.get("raw_actions") or [])
            raw.extend(seg.get("raw_actions") or [])
            prev["raw_actions"] = raw
            continue
        merged.append(dict(seg))
    return merged


def _refresh_segment_window(
    seg: dict,
    transcript: dict,
    visual_enriched: list,
) -> dict:
    """Re-pull transcript and visual fields for the segment's current time window."""
    start_sec = float(seg["start_sec"])
    end_sec = float(seg["end_sec"])
    timed_visual = _visual_timed_for_window(visual_enriched, start_sec, end_sec)
    observations = _dedupe_actions([v["action"] for v in timed_visual])
    if not observations:
        observations = _usable_observations(seg.get("visual_observations") or [])
    updated = dict(seg)
    updated.update(
        {
            "start_sec": round(start_sec, 3),
            "end_sec": round(end_sec, 3),
            "start_label": seconds_to_timestamp(start_sec),
            "end_label": seconds_to_timestamp(end_sec),
            "visual_observations": observations,
            "spoken_narration": _transcript_for_window(transcript, start_sec, end_sec),
            "timed_speech": _transcript_timed_for_window(transcript, start_sec, end_sec),
            "timed_visual": timed_visual,
        }
    )
    return updated


def filter_unimportant_segments(segments: list[dict]) -> list[dict]:
    kept = [s for s in segments if _is_important_segment(s)]
    skipped = len(segments) - len(kept)
    if skipped:
        print(f"  Skipped {skipped} unimportant segment(s) (idle / no actionable content).")
    return kept


def filter_unimportant_segments_preserving_coverage(
    segments: list[dict],
    transcript: dict,
    visual_enriched: list,
    video_duration_sec: float,
) -> list[dict]:
    """
    Drop segments with no relevant speech or visual content, but extend neighboring
    segments so the full uploaded video timeline stays covered.
    """
    if not segments:
        return []

    ordered = sorted(segments, key=lambda s: float(s["start_sec"]))
    kept: list[dict] = []
    skipped = 0

    for seg in ordered:
        if _is_important_segment(seg):
            kept.append(dict(seg))
            continue

        skipped += 1
        print(
            f"  Skipped unimportant segment "
            f"{seg.get('start_label', '?')}–{seg.get('end_label', '?')} "
            f"(idle / no actionable content); extending coverage."
        )
        if kept:
            kept[-1]["end_sec"] = float(seg["end_sec"])
            kept[-1] = _refresh_segment_window(kept[-1], transcript, visual_enriched)

    if skipped:
        print(f"  Skipped {skipped} unimportant segment(s) while preserving video coverage.")

    if kept and float(kept[-1]["end_sec"]) < video_duration_sec:
        kept[-1]["end_sec"] = round(video_duration_sec, 3)
        kept[-1] = _refresh_segment_window(kept[-1], transcript, visual_enriched)

    return kept


def fill_timeline_gaps(
    segments: list[dict],
    transcript: dict,
    visual_enriched: list,
    video_duration_sec: float,
) -> list[dict]:
    """Insert segments for timeline gaps that have transcript or visual content."""
    video_end = round(video_duration_sec, 3)
    if not segments:
        return []

    ordered = sorted(segments, key=lambda s: float(s["start_sec"]))
    filled: list[dict] = []
    cursor = 0.0

    for seg in ordered:
        start = float(seg["start_sec"])
        if start > cursor + MIN_SPEECH_SPLIT_SECONDS:
            gap_entry = _make_segment_entry(
                {
                    "start_sec": round(cursor, 3),
                    "end_sec": round(start, 3),
                    "actions": [],
                },
                transcript,
                visual_enriched,
            )
            if _is_important_segment(gap_entry):
                filled.append(gap_entry)
        filled.append(seg)
        cursor = max(cursor, float(seg["end_sec"]))

    if video_end > cursor + MIN_SPEECH_SPLIT_SECONDS:
        tail_entry = _make_segment_entry(
            {
                "start_sec": round(cursor, 3),
                "end_sec": video_end,
                "actions": [],
            },
            transcript,
            visual_enriched,
        )
        if _is_important_segment(tail_entry):
            filled.append(tail_entry)
    elif video_end > cursor and filled:
        filled[-1] = dict(filled[-1])
        filled[-1]["end_sec"] = video_end
        filled[-1] = _refresh_segment_window(filled[-1], transcript, visual_enriched)

    return filled


def renumber_segments(segments: list[dict]) -> list[dict]:
    for i, seg in enumerate(segments, start=1):
        seg["step_index"] = i
    return segments


def merge_visual_segments(
    visual_sequence: list,
    frame_interval: float,
    *,
    min_seconds: float = MIN_STEP_SECONDS,
    max_seconds: float = MAX_STEP_SECONDS,
) -> list[dict]:
    """Merge frame-level visual actions into non-overlapping time windows for clips."""
    enriched = enrich_visual_sequence(visual_sequence, frame_interval)
    if not enriched:
        return []

    merged: list[dict] = []
    bucket_actions: list[str] = []
    bucket_start = enriched[0]["start_sec"]
    bucket_end = enriched[0]["end_sec"]
    bucket_actions.append(enriched[0].get("action", ""))

    for seg in enriched[1:]:
        seg_start = seg["start_sec"]
        seg_end = seg["end_sec"]
        prospective_duration = seg_end - bucket_start
        same_action = _actions_similar(bucket_actions[-1], seg.get("action", ""))

        if prospective_duration <= max_seconds and (same_action or prospective_duration < min_seconds):
            bucket_end = seg_end
            bucket_actions.append(seg.get("action", ""))
            continue

        if bucket_end > bucket_start:
            merged.append(
                {
                    "start_sec": round(bucket_start, 3),
                    "end_sec": round(bucket_end, 3),
                    "actions": _dedupe_actions(bucket_actions),
                }
            )

        bucket_start = seg_start
        bucket_end = seg_end
        bucket_actions = [seg.get("action", "")]

    if bucket_end > bucket_start:
        merged.append(
            {
                "start_sec": round(bucket_start, 3),
                "end_sec": round(bucket_end, 3),
                "actions": _dedupe_actions(bucket_actions),
            }
        )

    return merged


def _transcript_for_window(transcript: dict, start_sec: float, end_sec: float) -> str:
    parts = [item["text"] for item in _transcript_timed_for_window(transcript, start_sec, end_sec)]
    return " ".join(parts).strip()


def _transcript_timed_for_window(
    transcript: dict, start_sec: float, end_sec: float
) -> list[dict]:
    items: list[dict] = []
    for segment in transcript.get("segments", []):
        seg_start = float(segment.get("start", 0) or 0)
        seg_end = float(segment.get("end", seg_start) or seg_start)
        if not _overlaps_time(start_sec, end_sec, seg_start, seg_end):
            continue
        text = (segment.get("text") or "").strip()
        if not text:
            continue
        items.append(
            {
                "start_sec": round(seg_start, 3),
                "end_sec": round(seg_end, 3),
                "start_label": seconds_to_timestamp(seg_start),
                "end_label": seconds_to_timestamp(seg_end),
                "text": text,
            }
        )
    return items


def _visual_timed_for_window(
    visual_enriched: list, start_sec: float, end_sec: float
) -> list[dict]:
    items: list[dict] = []
    for step in visual_enriched:
        seg_start = float(step.get("start_sec", 0))
        seg_end = float(step.get("end_sec", seg_start))
        if not _overlaps_time(start_sec, end_sec, seg_start, seg_end):
            continue
        action = (step.get("action") or "").strip()
        if not action or _is_idle_action(action):
            continue
        items.append(
            {
                "start_sec": seg_start,
                "end_sec": seg_end,
                "start_label": step.get("start_label", seconds_to_timestamp(seg_start)),
                "end_label": step.get("end_label", seconds_to_timestamp(seg_end)),
                "action": action,
            }
        )
    return items


def _split_text_by_enumeration(
    text: str, start_sec: float, end_sec: float
) -> list[dict]:
    """Split narration at first/second/third... and estimate sub-window timings."""
    parts = [p.strip() for p in ENUMERATION_SPLIT_RE.split(text) if p.strip()]
    if len(parts) <= 1:
        return [{"start_sec": start_sec, "end_sec": end_sec, "text": text.strip()}]

    duration = max(end_sec - start_sec, MIN_SPEECH_SPLIT_SECONDS)
    total_chars = sum(len(p) for p in parts) or 1
    cursor = start_sec
    chunks: list[dict] = []
    for part in parts:
        part_duration = duration * (len(part) / total_chars)
        part_end = min(end_sec, cursor + part_duration)
        if part_end <= cursor:
            part_end = min(end_sec, cursor + MIN_SPEECH_SPLIT_SECONDS)
        chunks.append(
            {
                "start_sec": round(cursor, 3),
                "end_sec": round(part_end, 3),
                "text": part,
            }
        )
        cursor = part_end
    if chunks:
        chunks[-1]["end_sec"] = round(end_sec, 3)
    return chunks


def _split_segment_by_speech(segment: dict) -> list[dict]:
    """Split a segment when narration enumerates items or has multiple speech beats."""
    duration = segment["end_sec"] - segment["start_sec"]
    timed_speech = segment.get("timed_speech") or []
    speech_chunks: list[dict] = []

    if len(timed_speech) > 1:
        for part in timed_speech:
            start = max(segment["start_sec"], part["start_sec"])
            end = min(segment["end_sec"], part["end_sec"])
            if end > start:
                speech_chunks.append(
                    {"start_sec": start, "end_sec": end, "text": part["text"]}
                )
    elif timed_speech and _has_enumeration(timed_speech[0].get("text", "")):
        speech_chunks = _split_text_by_enumeration(
            timed_speech[0]["text"],
            segment["start_sec"],
            segment["end_sec"],
        )
    elif segment.get("spoken_narration") and _has_enumeration(segment["spoken_narration"]):
        speech_chunks = _split_text_by_enumeration(
            segment["spoken_narration"],
            segment["start_sec"],
            segment["end_sec"],
        )
    elif duration > MAX_STEP_SECONDS and len(timed_speech) > 1:
        for part in timed_speech:
            start = max(segment["start_sec"], part["start_sec"])
            end = min(segment["end_sec"], part["end_sec"])
            if end > start:
                speech_chunks.append(
                    {"start_sec": start, "end_sec": end, "text": part["text"]}
                )

    if len(speech_chunks) <= 1:
        return [segment]

    splits: list[dict] = []
    timed_visual_all = segment.get("timed_visual") or []
    for chunk in speech_chunks:
        start = chunk["start_sec"]
        end = chunk["end_sec"]
        if end - start < MIN_SPEECH_SPLIT_SECONDS:
            continue
        timed_visual = _filter_timed_items(timed_visual_all, start, end)
        observations = _dedupe_actions([v["action"] for v in timed_visual])
        if not observations:
            observations = list(segment.get("visual_observations") or [])
        window_speech = _filter_timed_items(segment.get("timed_speech") or [], start, end)
        narration = " ".join(p["text"] for p in window_speech).strip() or chunk["text"]

        new_seg = dict(segment)
        new_seg.update(
            {
                "start_sec": round(start, 3),
                "end_sec": round(end, 3),
                "start_label": seconds_to_timestamp(start),
                "end_label": seconds_to_timestamp(end),
                "spoken_narration": narration,
                "timed_speech": window_speech
                or [
                    {
                        "start_sec": round(start, 3),
                        "end_sec": round(end, 3),
                        "start_label": seconds_to_timestamp(start),
                        "end_label": seconds_to_timestamp(end),
                        "text": chunk["text"],
                    }
                ],
                "timed_visual": timed_visual,
                "visual_observations": observations,
            }
        )
        splits.append(new_seg)

    return splits if splits else [segment]


def _filter_timed_items(items: list[dict], start_sec: float, end_sec: float) -> list[dict]:
    kept: list[dict] = []
    for item in items:
        item_start = float(item.get("start_sec", start_sec))
        item_end = float(item.get("end_sec", item_start))
        if _overlaps_time(start_sec, end_sec, item_start, item_end):
            kept.append(item)
    return kept


def refine_segments_by_speech(segments: list[dict]) -> list[dict]:
    """Split coarse visual windows into clips aligned to spoken beats."""
    refined: list[dict] = []
    for segment in segments:
        refined.extend(_split_segment_by_speech(segment))
    if len(refined) > len(segments):
        print(
            f"  Speech-aligned split: {len(segments)} visual window(s) -> "
            f"{len(refined)} clip segment(s)."
        )
    return refined


def _make_segment_entry(
    seg: dict,
    transcript: dict,
    visual_enriched: list,
) -> dict:
    start_sec = seg["start_sec"]
    end_sec = seg["end_sec"]
    timed_visual = _visual_timed_for_window(visual_enriched, start_sec, end_sec)
    return {
        "start_sec": start_sec,
        "end_sec": end_sec,
        "start_label": seconds_to_timestamp(start_sec),
        "end_label": seconds_to_timestamp(end_sec),
        "visual_observations": _usable_observations(seg["actions"]),
        "raw_actions": seg.get("actions") or [],
        "spoken_narration": _transcript_for_window(transcript, start_sec, end_sec),
        "timed_speech": _transcript_timed_for_window(transcript, start_sec, end_sec),
        "timed_visual": timed_visual,
    }


def _extract_wi_fields(item: dict) -> tuple[str, list[str], list[str]]:
    wi = (
        item.get("WI")
        or item.get("wi")
        or item.get("work_instruction")
        or item.get("instruction")
        or item.get("description")
        or item.get("text")
        or ""
    )
    wi = str(wi).strip()
    instruments = item.get("instruments") or item.get("tools") or []
    if isinstance(instruments, str):
        instruments = [p.strip() for p in re.split(r"[,;]", instruments) if p.strip()]
    subitems = item.get("subitems") or item.get("steps") or item.get("items") or []
    if isinstance(subitems, str):
        subitems = [p.strip() for p in re.split(r"[\n;]", subitems) if p.strip()]
    subitems = [str(s).strip() for s in subitems if str(s).strip()]
    return wi, instruments, subitems


def _is_weak_wi(text: str) -> bool:
    text = (text or "").strip()
    if len(text) < 25:
        return True
    normalized = text.lower().rstrip(".") + "."
    return normalized in WEAK_WI_EXACT


def _polish_wi_text(text: str) -> str:
    text = (text or "").strip()
    for pattern in META_WI_PATTERNS:
        text = pattern.sub("", text).strip()
    text = INLINE_TIMESTAMP_RE.sub("", text)
    text = re.sub(r"\s*---+\s*", " ", text)
    text = re.sub(r"#{1,6}\s+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if text and not text.endswith("."):
        text += "."
    return text


def _join_ordinals(words: list[str]) -> str:
    if not words:
        return ""
    if len(words) == 1:
        return words[0]
    if len(words) == 2:
        return f"{words[0]} and {words[1]}"
    return f"{', '.join(words[:-1])}, and {words[-1]}"


def _sanitize_subitems(subitems: list[str]) -> list[str]:
    """Keep only short enumerated steps; drop per-second frame breakdown lines."""
    cleaned: list[str] = []
    for item in subitems:
        item = (item or "").strip()
        if not item:
            continue
        if TIMESTAMP_SUBITEM_RE.match(item):
            continue
        if len(item) > 220:
            continue
        cleaned.append(item.rstrip(".").strip() + ".")
    return cleaned


def _merge_subitems_into_wi(wi: str, subitems: list[str]) -> str:
    """Fold any list lines into a single WI string."""
    wi = _polish_wi_text(wi)
    extras = [s.strip().rstrip(".") for s in subitems if (s or "").strip()]
    if not extras:
        return wi
    if wi:
        return _polish_wi_text(f"{wi.rstrip('.')}. " + ". ".join(extras) + ".")
    return _polish_wi_text(". ".join(extras) + ".")


def _finalize_step_wi(
    segment: dict, wi: str, instruments: list[str], subitems: list[str]
) -> tuple[str, list[str], list[str]]:
    speech = (segment.get("spoken_narration") or "").strip()

    subitems = _sanitize_subitems(subitems)
    wi = _merge_subitems_into_wi(wi, subitems)

    if _is_weak_wi(wi) or any(p.search(wi) for p in META_WI_PATTERNS):
        wi, instruments, _ = _heuristic_wi(segment)
        wi = _polish_wi_text(wi)

    if speech and len(wi) < 40:
        wi = _polish_wi_text(_heuristic_wi(segment)[0])

    wi = _polish_wi_text(wi)
    return wi, instruments, []


def _heuristic_wi(segment: dict) -> tuple[str, list[str], list[str]]:
    """Build simple imperative work instructions in one WI string."""
    observations = _usable_observations(segment.get("visual_observations") or [])
    speech = (segment.get("spoken_narration") or "").strip()
    instruments: list[str] = []

    for obs in observations:
        for tool in ("flashlight", "screwdriver", "wrench", "multimeter", "pliers"):
            if tool in obs.lower() and tool not in instruments:
                instruments.append(tool)

    if speech:
        for tool in ("flashlight", "screwdriver", "tie-down", "strap", "checklist"):
            if tool in speech.lower() and tool not in instruments:
                instruments.append(tool)

    if _has_enumeration(speech):
        ordinal_tokens = [m.group(1).lower() for m in ORDINALS_RE.finditer(speech)]
        ordinal_tokens = list(dict.fromkeys(ordinal_tokens))
        combined_obs = " ".join(observations).lower()
        if len(ordinal_tokens) >= 2 and ("optic" in speech.lower() or "optic" in combined_obs):
            ordinal_phrase = _join_ordinals(ordinal_tokens)
            wi = f"Inspect the {ordinal_phrase} optics within the assembly."
            return _polish_wi_text(wi), instruments, []

        enum_parts = [
            p.strip().rstrip(".") for p in ENUMERATION_SPLIT_RE.split(speech) if p.strip()
        ]
        lines: list[str] = []
        for part in enum_parts:
            part = part[0].upper() + part[1:] if part else part
            if not part.lower().startswith(
                ("inspect", "check", "verify", "secure", "route", "mark", "confirm")
            ):
                part = f"Inspect {part.lstrip()}"
            lines.append(f"{part.rstrip('.')}.")
        wi = " ".join(lines)
    elif speech:
        wi = speech[0].upper() + speech[1:] if speech else speech
        if not wi.endswith("."):
            wi += "."
        if observations:
            visual = observations[0].rstrip(".")
            if visual.lower() not in wi.lower():
                wi = f"{wi.rstrip('.')}. {visual}."
    elif observations:
        wi = ". ".join(o.rstrip(".") for o in observations) + "."
    else:
        wi = "Perform the work shown in this step."

    return _polish_wi_text(wi), instruments, []


def _generate_step_wi_gpt(segment: dict, prior_wi_summaries: list[str] | None = None) -> tuple[str, list[str], list[str]]:
    """One focused GPT call per clip segment for operator-ready instructions."""
    client = get_client()
    model = get_writer_model()

    visual = _usable_observations(segment.get("visual_observations") or [])
    speech = (segment.get("spoken_narration") or "").strip() or "(no narration in this segment)"

    prompt = (
        f"Write work instructions for this video segment only "
        f"({segment['start_label']} to {segment['end_label']}).\n\n"
        f"SPOKEN NARRATION (preserve exact wording when present):\n{speech}\n\n"
        f"VISIBLE ACTIONS IN THIS SEGMENT:\n"
        f"{json.dumps(visual, ensure_ascii=False)}\n"
        "Output one WI field only — a single string the operator can read and follow.\n"
        "Rules:\n"
        "- Base WI ONLY on the spoken narration and visible actions above.\n"
        "- Keep narrator terms, but rewrite only enough to fix fragmented speech into natural operator wording.\n"
        "- Fuse narration intent with the on-screen action into one coherent instruction.\n"
        "- Add a visible action only if the narrator did not say it and it appears on screen.\n"
        "- If narration has split enumeration fragments (e.g. 'third four'), convert to natural phrasing grounded in visible objects.\n"
        "- Do not output alternatives like 'or inspect'; commit to the action supported by narration+visual evidence.\n"
        "- Do NOT add titles, safety notes, explanations, or content not in the inputs.\n"
        "- Do NOT reference the video, clip, segment, or timestamps.\n"
        "- Plain imperative sentences only; no bullets, lists, or markdown.\n"
        "- If the narrator lists multiple items, include each exactly as spoken, as separate sentences.\n"
        'Return JSON only: {"WI": "your work instructions here.", "instruments": []}'
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You tie spoken narration and visible actions into operator work "
                    "instructions. Use only the provided inputs; do not invent facts "
                    "or describe the recording. Resolve disfluencies into clear, faithful WI."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
        max_tokens=320,
    )

    raw = response.choices[0].message.content or "{}"
    item = json.loads(raw)
    wi, instruments, subitems = _extract_wi_fields(item)
    wi, instruments, subitems = _finalize_step_wi(segment, wi, instruments, subitems)
    if _is_weak_wi(wi):
        return _heuristic_wi(segment)
    return wi, instruments, subitems


def _build_step_from_segment(segment: dict, prior_wi_summaries: list[str] | None = None) -> dict:
    try:
        wi, instruments, subitems = _generate_step_wi_gpt(segment, prior_wi_summaries)
    except Exception as exc:
        print(f"WARNING: step {segment.get('step_index')} GPT failed ({exc}); using heuristic.")
        wi, instruments, subitems = _heuristic_wi(segment)

    wi, instruments, subitems = _finalize_step_wi(segment, wi, instruments, subitems)

    return {
        "text": wi,
        "start_sec": segment["start_sec"],
        "end_sec": segment["end_sec"],
        "start_label": segment["start_label"],
        "end_label": segment["end_label"],
        "instruments": instruments,
        "subitems": [],
    }


def prepare_segment_payload(
    visual_sequence: list,
    transcript: dict,
    frame_interval: float,
    video_duration_sec: float,
) -> list[dict]:
    """Build non-overlapping segment windows (no GPT). Used before streaming WI generation."""
    visual_enriched = enrich_visual_sequence(visual_sequence, frame_interval)
    merged = merge_visual_segments(visual_sequence, frame_interval)
    if not merged:
        return []

    merged[-1]["end_sec"] = round(video_duration_sec, 3)

    segment_payload: list[dict] = []
    for seg in merged:
        segment_payload.append(_make_segment_entry(seg, transcript, visual_enriched))

    segment_payload = refine_segments_by_speech(segment_payload)
    segment_payload = fill_timeline_gaps(
        segment_payload, transcript, visual_enriched, video_duration_sec
    )
    segment_payload = filter_unimportant_segments_preserving_coverage(
        segment_payload, transcript, visual_enriched, video_duration_sec
    )
    if segment_payload and float(segment_payload[0]["start_sec"]) > 0:
        segment_payload[0] = dict(segment_payload[0])
        segment_payload[0]["start_sec"] = 0.0
        segment_payload[0] = _refresh_segment_window(
            segment_payload[0], transcript, visual_enriched
        )
    segment_payload = renumber_segments(segment_payload)
    return segment_payload


def _clean_wi_step_body(body: str) -> str:
    """Normalize multi-line step prose from generated work instructions."""
    text = (body or "").strip()
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"\s*\n+\s*", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _merge_parsed_step_text(block: dict) -> str:
    """Combine main step body with any continuation lines from the parser."""
    text = (block.get("text") or "").strip()
    if text.lower() == WI_PLACEHOLDER_TEXT.lower():
        text = ""
    subitems = block.get("subitems") or []
    if not subitems:
        return text
    parts = [text] if text else []
    for item in subitems:
        item = (item or "").strip()
        if not item or item.lower() == WI_PLACEHOLDER_TEXT.lower():
            continue
        if not item.endswith("."):
            item += "."
        parts.append(item)
    return " ".join(parts).strip()


def _extract_timestamped_steps(
    work_instructions: str,
    video_duration_sec: float,
) -> list[dict]:
    """Extract numbered steps with [start - end] timestamps and multi-line bodies."""
    steps: list[dict] = []
    for match in WI_TIMESTAMPED_STEP_RE.finditer(work_instructions):
        start_sec = timestamp_label_to_seconds(
            match.group("start").strip(), video_duration_sec
        )
        end_sec = timestamp_label_to_seconds(
            match.group("end").strip(), video_duration_sec
        )
        body = _clean_wi_step_body(match.group("body"))
        if not body or body.lower() == WI_PLACEHOLDER_TEXT.lower():
            continue
        body, instruments = extract_instruments(body)
        if not body.strip() or _is_weak_wi(body):
            continue
        steps.append(
            {
                "text": body.strip(),
                "start_sec": round(start_sec, 3),
                "end_sec": round(end_sec, 3),
                "start_label": seconds_to_timestamp(start_sec),
                "end_label": seconds_to_timestamp(end_sec),
                "instruments": instruments,
                "subitems": [],
            }
        )
    return steps


def _steps_from_parse_blocks(
    work_instructions: str,
    video_duration_sec: float,
) -> list[dict]:
    """Fallback parser using report_builder block structure."""
    blocks = parse_work_instructions_blocks(work_instructions, video_duration_sec)
    steps: list[dict] = []
    pending: dict | None = None

    def _flush_pending() -> None:
        nonlocal pending
        if not pending:
            return
        text = _merge_parsed_step_text(pending)
        text, instruments = extract_instruments(text)
        if text.strip() and not _is_weak_wi(text):
            steps.append(
                {
                    "text": text.strip(),
                    "start_sec": float(pending["start_sec"]),
                    "end_sec": float(pending["end_sec"]),
                    "start_label": pending["start_label"],
                    "end_label": pending["end_label"],
                    "instruments": instruments,
                    "subitems": [],
                }
            )
        pending = None

    for block in blocks:
        if block.get("type") == "step":
            _flush_pending()
            pending = block
            continue
        if pending and block.get("type") == "paragraph":
            pending.setdefault("subitems", []).append(block.get("text", ""))
            continue
        _flush_pending()

    _flush_pending()
    return steps


def build_steps_from_work_instructions(
    work_instructions: str,
    video_duration_sec: float,
) -> list[dict]:
    """Parse full-document work instructions into ordered WS step dicts."""
    steps = _extract_timestamped_steps(work_instructions, video_duration_sec)
    if not steps:
        steps = _steps_from_parse_blocks(work_instructions, video_duration_sec)

    return enforce_non_overlapping_steps(steps, video_duration_sec)


def cut_parsed_wi_clips(
    steps: list[dict],
    video_path: str,
    clips_dir: str,
    *,
    max_workers: int = 8,
    start_upload: Callable[[int, str], Future | None] | None = None,
) -> list[tuple[int, dict, str | None]]:
    """
    Cut clips for parsed WI steps in parallel; return ordered (step_index, step, clip_path).
    """
    if not steps:
        return []

    os.makedirs(clips_dir, exist_ok=True)
    indexed_steps: list[dict] = []
    for i, step in enumerate(steps, start=1):
        indexed_steps.append({**step, "step_index": i})

    print(f"Cutting {len(indexed_steps)} clips from parsed work instructions...")
    workers = max(1, min(max_workers, len(indexed_steps)))
    clip_paths: dict[int, str | None] = {}

    def _cut_one(seg: dict) -> tuple[int, str | None]:
        seg_index = seg["step_index"]
        clip_path = os.path.join(clips_dir, f"segment_{seg_index:03d}.mp4")
        try:
            if cut_video_clip(video_path, clip_path, seg["start_sec"], seg["end_sec"]):
                return seg_index, clip_path
        except Exception as exc:
            print(f"WARNING: clip cut failed for step {seg_index}: {exc}")
        return seg_index, None

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_cut_one, seg) for seg in indexed_steps]
        for future in as_completed(futures):
            seg_index, clip_path = future.result()
            clip_paths[seg_index] = clip_path
            if clip_path and start_upload is not None:
                start_upload(seg_index, clip_path)

    prepared: list[tuple[int, dict, str | None]] = []
    for seg in indexed_steps:
        seg_index = seg["step_index"]
        clip_path = clip_paths.get(seg_index)
        step = {k: v for k, v in seg.items() if k != "step_index"}
        prepared.append((seg_index, step, clip_path))
        print(f"  Step {seg_index}/{len(indexed_steps)} ready (parsed WI + clip).")

    return prepared


def stream_aligned_work_steps(
    segment_payload: list[dict],
    video_duration_sec: float,
    on_step_ready,
    *,
    max_workers: int = 5,
    ordered: bool = True,
) -> list[dict]:
    """
    Generate WI per segment in parallel.
    on_step_ready(segment_step_index, step) is called when each segment completes.
    If ordered=True, the caller should buffer and emit in sequence (see OrderedStepEmitter).
    """
    if not segment_payload:
        return []

    print(f"Generating {len(segment_payload)} work-instruction steps...")
    steps: list[dict | None] = [None] * len(segment_payload)
    workers = max(1, min(max_workers, len(segment_payload)))
    prior_wi_lock = threading.Lock()
    prior_wi: list[str] = []

    def _build_with_context(seg: dict) -> dict:
        with prior_wi_lock:
            context = list(prior_wi)
        step = _build_step_from_segment(seg, context)
        with prior_wi_lock:
            prior_wi.append(step.get("text", ""))
        return step

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_build_with_context, seg): i
            for i, seg in enumerate(segment_payload)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                step = future.result()
            except Exception as exc:
                print(f"  Step {idx + 1} failed ({exc}); using heuristic.")
                step = _build_step_from_segment_heuristic_only(segment_payload[idx])

            steps[idx] = step
            seg_index = segment_payload[idx]["step_index"]
            on_step_ready(seg_index, step)
            print(f"  Segment {seg_index}/{len(segment_payload)} WI generated.")

    finalized = enforce_non_overlapping_steps(
        [s for s in steps if s is not None],
        video_duration_sec,
        segment_payload,
    )
    return finalized


def stream_segments_with_clips(
    segment_payload: list[dict],
    video_path: str,
    clips_dir: str,
    video_duration_sec: float,
    on_step_ready,
    *,
    max_workers: int = 8,
    start_upload: Callable[[int, str], Future | None] | None = None,
) -> list[dict]:
    """
    Cut clips in parallel, then generate WI sequentially so each step sees prior steps.
    on_step_ready(segment_step_index, step, clip_path) fires as each segment completes.
    """
    if not segment_payload:
        return []

    os.makedirs(clips_dir, exist_ok=True)
    print(
        f"Processing {len(segment_payload)} segments "
        f"(parallel clip cut, sequential WI generation)..."
    )
    clip_paths: dict[int, str | None] = {}
    workers = max(1, min(max_workers, len(segment_payload)))

    def _cut_clip(seg: dict) -> tuple[int, str | None]:
        seg_index = seg["step_index"]
        clip_path = os.path.join(clips_dir, f"segment_{seg_index:03d}.mp4")
        try:
            cut_video_clip(video_path, clip_path, seg["start_sec"], seg["end_sec"])
            return seg_index, clip_path
        except Exception as exc:
            print(f"WARNING: clip cut failed for segment {seg_index}: {exc}")
            return seg_index, None

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_cut_clip, seg) for seg in segment_payload]
        for future in as_completed(futures):
            seg_index, clip_path = future.result()
            clip_paths[seg_index] = clip_path
            if clip_path and start_upload is not None:
                start_upload(seg_index, clip_path)

    steps: list[dict] = []
    prior_wi: list[str] = []
    for seg in segment_payload:
        seg_index = seg["step_index"]
        clip_path = clip_paths.get(seg_index)
        try:
            step = _build_step_from_segment(seg, prior_wi)
        except Exception as exc:
            print(f"WARNING: segment {seg_index} WI failed ({exc}); using heuristic.")
            step = _build_step_from_segment_heuristic_only(seg)

        wi = (step.get("text") or "").strip()
        steps.append(step)
        prior_wi.append(wi)
        on_step_ready(seg_index, step, clip_path)
        print(f"  Segment {seg_index}/{len(segment_payload)} ready (clip + WI).")

    return enforce_non_overlapping_steps(
        steps,
        video_duration_sec,
        segment_payload,
    )


def generate_aligned_work_steps(
    visual_sequence: list,
    transcript: dict,
    frame_interval: float,
    video_duration_sec: float,
    *,
    max_workers: int = 5,
) -> list[dict]:
    """
    Build non-overlapping WI steps aligned to visual timeline segments.
    Each step gets its own GPT call so WI matches the clip content.
    """
    segment_payload = prepare_segment_payload(
        visual_sequence, transcript, frame_interval, video_duration_sec
    )
    if not segment_payload:
        return []

    print(f"Generating {len(segment_payload)} aligned work-instruction steps...")
    steps: list[dict | None] = [None] * len(segment_payload)
    workers = max(1, min(max_workers, len(segment_payload)))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_build_step_from_segment, seg): i
            for i, seg in enumerate(segment_payload)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                steps[idx] = future.result()
                print(f"  Step {idx + 1}/{len(segment_payload)} WI ready.")
            except Exception as exc:
                print(f"  Step {idx + 1} failed ({exc}); using heuristic.")
                steps[idx] = _build_step_from_segment_heuristic_only(segment_payload[idx])

    return enforce_non_overlapping_steps(
        [s for s in steps if s is not None],
        video_duration_sec,
        segment_payload,
    )


def _build_step_from_segment_heuristic_only(segment: dict) -> dict:
    wi, instruments, subitems = _heuristic_wi(segment)
    return {
        "text": wi,
        "start_sec": segment["start_sec"],
        "end_sec": segment["end_sec"],
        "start_label": segment["start_label"],
        "end_label": segment["end_label"],
        "instruments": instruments,
        "subitems": [],
    }


def enforce_non_overlapping_steps(
    steps: list[dict],
    video_duration_sec: float,
    segments: list[dict] | None = None,
) -> list[dict]:
    """Sequential, non-overlapping clip windows. Never replace WI with generic text."""
    if not steps:
        return []

    ordered = sorted(steps, key=lambda s: float(s.get("start_sec", 0)))
    fixed: list[dict] = []
    cursor = 0.0

    for i, step in enumerate(ordered):
        start = max(cursor, float(step.get("start_sec", cursor)))
        end = float(step.get("end_sec", start))
        end = min(end, video_duration_sec)

        if end <= start:
            end = min(start + 2.0, video_duration_sec)
        if end <= start:
            continue

        text = (step.get("text") or "").strip()
        subitems = step.get("subitems") or []
        if _is_weak_wi(text) and segments and i < len(segments):
            text, instruments, _ = _heuristic_wi(segments[i])
        else:
            instruments = step.get("instruments") or []

        fixed.append(
            {
                "text": text,
                "start_sec": round(start, 3),
                "end_sec": round(end, 3),
                "start_label": seconds_to_timestamp(start),
                "end_label": seconds_to_timestamp(end),
                "instruments": instruments,
                "subitems": [],
            }
        )
        cursor = end

    return fixed
