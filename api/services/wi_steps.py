import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher

from azure_client import get_client, get_writer_model
from report_builder import enrich_visual_sequence
from time_utils import seconds_to_timestamp

MIN_STEP_SECONDS = 8.0
MAX_STEP_SECONDS = 32.0
WI_SIMILARITY_THRESHOLD = 0.68
MIN_SPEECH_CHARS = 12

WEAK_WI_EXACT = frozenset(
    {
        "complete the task shown in this video segment.",
        "view this segment in the video.",
        "no action.",
    }
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


def _segment_similar(a: dict, b: dict) -> bool:
    """True when two adjacent segments describe essentially the same work."""
    a_obs = " ".join(a.get("visual_observations") or [])
    b_obs = " ".join(b.get("visual_observations") or [])
    a_speech = a.get("spoken_narration") or ""
    b_speech = b.get("spoken_narration") or ""

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
            raw = list(prev.get("raw_actions") or [])
            raw.extend(seg.get("raw_actions") or [])
            prev["raw_actions"] = raw
            continue
        merged.append(dict(seg))
    return merged


def filter_unimportant_segments(segments: list[dict]) -> list[dict]:
    kept = [s for s in segments if _is_important_segment(s)]
    skipped = len(segments) - len(kept)
    if skipped:
        print(f"  Skipped {skipped} unimportant segment(s) (idle / no actionable content).")
    return kept


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
    parts: list[str] = []
    for segment in transcript.get("segments", []):
        seg_start = float(segment.get("start", 0) or 0)
        seg_end = float(segment.get("end", seg_start) or seg_start)
        if seg_end <= start_sec or seg_start >= end_sec:
            continue
        text = (segment.get("text") or "").strip()
        if text:
            parts.append(text)
    return " ".join(parts).strip()


def _extract_wi_fields(item: dict) -> tuple[str, list[str]]:
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
    return wi, instruments


def _is_weak_wi(text: str) -> bool:
    text = (text or "").strip()
    if len(text) < 25:
        return True
    normalized = text.lower().rstrip(".") + "."
    return normalized in WEAK_WI_EXACT


def _heuristic_wi(segment: dict) -> tuple[str, list[str]]:
    """Build imperative instructions from visual + spoken data without generic placeholders."""
    observations = _usable_observations(segment.get("visual_observations") or [])
    speech = (segment.get("spoken_narration") or "").strip()
    instruments: list[str] = []

    for obs in observations:
        for tool in ("flashlight", "screwdriver", "wrench", "multimeter", "pliers"):
            if tool in obs.lower():
                instruments.append(tool)

    if speech:
        for tool in ("flashlight", "screwdriver", "tie-down", "strap"):
            if tool in speech.lower() and tool not in instruments:
                instruments.append(tool)

    if speech and observations:
        primary = observations[0]
        wi = (
            f"During this segment, {primary.rstrip('.')}. "
            f"Follow the operator guidance: {speech.rstrip('.')}. "
            "Confirm the work matches quality standards before moving on."
        )
    elif speech:
        wi = (
            f"In this segment, follow the operator guidance: {speech.rstrip('.')}. "
            "Perform each check carefully and confirm the result before proceeding."
        )
    elif observations:
        primary = observations[0].rstrip(".")
        wi = f"In this segment, {primary}."
        if len(observations) > 1:
            wi += f" Also verify that {observations[1].rstrip('.').lower()}."
        wi += " Complete this activity before advancing to the next step."
    else:
        wi = (
            f"Review the assembly work shown between {segment['start_label']} and "
            f"{segment['end_label']}. Identify the task being performed and repeat it "
            "following the same method and safety practices."
        )

    return wi, instruments


def _generate_step_wi_gpt(segment: dict, prior_wi_summaries: list[str] | None = None) -> tuple[str, list[str]]:
    """One focused GPT call per clip segment for operator-ready instructions."""
    client = get_client()
    model = get_writer_model()

    visual = _usable_observations(segment.get("visual_observations") or [])
    speech = (segment.get("spoken_narration") or "").strip() or "(no narration in this segment)"

    prior_block = ""
    if prior_wi_summaries:
        prior_block = (
            "\nALREADY COVERED in earlier steps (do NOT repeat the same instructions):\n"
            + "\n".join(f"- {w[:200]}" for w in prior_wi_summaries[-3:])
            + "\n"
        )

    prompt = (
        f"You are writing work instructions for ONE short video clip: "
        f"{segment['start_label']} to {segment['end_label']}.\n\n"
        f"VISUAL (what happens in the clip):\n{json.dumps(visual, ensure_ascii=False)}\n\n"
        f"SPOKEN NARRATION:\n{speech}\n"
        f"{prior_block}\n"
        "Write 2-3 sentences telling the operator exactly what to DO in this clip. "
        "Use imperative verbs (Inspect, Verify, Check, Secure, Route, etc.). "
        "Combine what is seen and what is said into clear instructions.\n"
        "Do NOT repeat instructions already covered in earlier steps.\n"
        "Do NOT write: 'view this segment', 'complete the task shown', or 'worker is not visible'.\n"
        'Return JSON only: {"WI": "your instructions here", "instruments": ["tool1"]}'
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You write concise, actionable factory work instructions for a single video clip. "
                    "Every sentence tells the operator what to do."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
        max_tokens=280,
    )

    raw = response.choices[0].message.content or "{}"
    item = json.loads(raw)
    wi, instruments = _extract_wi_fields(item)
    if _is_weak_wi(wi):
        return _heuristic_wi(segment)
    return wi, instruments


def _build_step_from_segment(segment: dict, prior_wi_summaries: list[str] | None = None) -> dict:
    try:
        wi, instruments = _generate_step_wi_gpt(segment, prior_wi_summaries)
    except Exception as exc:
        print(f"WARNING: step {segment.get('step_index')} GPT failed ({exc}); using heuristic.")
        wi, instruments = _heuristic_wi(segment)

    if _is_weak_wi(wi):
        wi, instruments = _heuristic_wi(segment)

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
    merged = merge_visual_segments(visual_sequence, frame_interval)
    if not merged:
        return []

    merged[-1]["end_sec"] = min(merged[-1]["end_sec"], round(video_duration_sec, 3))

    segment_payload: list[dict] = []
    for seg in merged:
        segment_payload.append(
            {
                "start_sec": seg["start_sec"],
                "end_sec": seg["end_sec"],
                "start_label": seconds_to_timestamp(seg["start_sec"]),
                "end_label": seconds_to_timestamp(seg["end_sec"]),
                "visual_observations": _usable_observations(seg["actions"]),
                "raw_actions": seg.get("actions") or [],
                "spoken_narration": _transcript_for_window(
                    transcript, seg["start_sec"], seg["end_sec"]
                ),
            }
        )

    segment_payload = filter_unimportant_segments(segment_payload)
    segment_payload = merge_adjacent_similar_segments(segment_payload)
    segment_payload = renumber_segments(segment_payload)
    return segment_payload


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
    merged = merge_visual_segments(visual_sequence, frame_interval)
    if not merged:
        return []

    merged[-1]["end_sec"] = min(merged[-1]["end_sec"], round(video_duration_sec, 3))

    segment_payload: list[dict] = []
    for seg in merged:
        segment_payload.append(
            {
                "start_sec": seg["start_sec"],
                "end_sec": seg["end_sec"],
                "start_label": seconds_to_timestamp(seg["start_sec"]),
                "end_label": seconds_to_timestamp(seg["end_sec"]),
                "visual_observations": _usable_observations(seg["actions"]),
                "raw_actions": seg.get("actions") or [],
                "spoken_narration": _transcript_for_window(
                    transcript, seg["start_sec"], seg["end_sec"]
                ),
            }
        )

    segment_payload = filter_unimportant_segments(segment_payload)
    segment_payload = merge_adjacent_similar_segments(segment_payload)
    segment_payload = renumber_segments(segment_payload)
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
    wi, instruments = _heuristic_wi(segment)
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
        if _is_weak_wi(text) and segments and i < len(segments):
            text, instruments = _heuristic_wi(segments[i])
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
                "subitems": step.get("subitems") or [],
            }
        )
        cursor = end

    return fixed
