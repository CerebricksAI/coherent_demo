import re

INSTRUMENTS_LINE_RE = re.compile(
    r"(?:^|\n)\s*(?:[-*]\s*)?(?:\*\*)?Instruments used:\s*(.+?)(?:\*\*)?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def extract_instruments(step_text: str) -> tuple[str, list[str]]:
    """Parse 'Instruments used:' line from step text; return clean WI + instrument list."""
    match = INSTRUMENTS_LINE_RE.search(step_text)
    if not match:
        return step_text.strip(), []

    instruments_raw = match.group(1).strip()
    instruments_raw = re.sub(r"\*\*", "", instruments_raw)
    instruments = [p.strip() for p in re.split(r"[,;]", instruments_raw) if p.strip()]

    clean = INSTRUMENTS_LINE_RE.sub("", step_text).strip()
    clean = re.sub(r"\n{3,}", "\n\n", clean)
    return clean, instruments


def build_step_event(
    job_id: str,
    step_index: int,
    step: dict,
    video_clip_url: str | None,
) -> dict:
    wi_text = (step.get("text") or "").strip()
    instruments = step.get("instruments") or []
    if not instruments:
        wi_text, instruments = extract_instruments(wi_text)

    if step.get("subitems"):
        subitems = "\n".join(f"- {s}" for s in step["subitems"])
        wi_text = f"{wi_text}\n{subitems}".strip()

    return {
        "type": "step",
        "job_id": job_id,
        "step_index": step_index,
        "WI": wi_text,
        "timestamp_start": step.get("start_label", ""),
        "timestamp_end": step.get("end_label", ""),
        "video_clip_url": video_clip_url,
        "instruments": instruments,
    }
