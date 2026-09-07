import base64
import html
import re
from difflib import SequenceMatcher

from azure_client import get_client, get_gpt4o_model

FRAME_PROMPT = (
    "You are an industrial process analyst reviewing one frame from a factory worker video.\n"
    "Describe ONLY what the worker is physically doing with their hands and tools (1-2 sentences).\n"
    "Focus on assembly, inspection, tightening fasteners, routing cables, using a flashlight, etc.\n\n"
    "Rules:\n"
    "- Do NOT transcribe on-screen text, labels, documents, or part numbers\n"
    "- Do NOT use markdown, tables, bullet lists, or image references\n"
    "- Plain English sentences only"
)

RETRY_PROMPT = (
    "Describe the worker's physical action in this factory video frame in one plain sentence. "
    "Ignore all visible text and labels. No markdown."
)

_GARBAGE_PATTERNS = (
    re.compile(r"!\[.*?\]\([^)]+\)"),  # markdown images
    re.compile(r"\| --- \|"),  # markdown tables
    re.compile(r"img-\d+\.(jpe?g|png)", re.I),
    re.compile(r"^#\s*[\d\s]+$", re.M),
)


def is_garbage_vision_response(text: str) -> bool:
    """Detect OCR-style or markdown junk instead of action descriptions."""
    text = (text or "").strip()
    if len(text) < 12:
        return True
    for pattern in _GARBAGE_PATTERNS:
        if pattern.search(text):
            return True
    digits = sum(c.isdigit() for c in text)
    if digits / max(len(text), 1) > 0.45:
        return True
    if text.count("|") >= 4 and "---" in text:
        return True
    return False


def _clean_vision_text(text: str) -> str:
    text = html.unescape((text or "").strip())
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def describe_frame(image_path: str) -> str:
    """Analyze a single frame with GPT-4o vision."""
    client = get_client()
    model = get_gpt4o_model()

    with open(image_path, "rb") as image_file:
        b64 = base64.b64encode(image_file.read()).decode("utf-8")

    image_part = {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
    }

    for prompt in (FRAME_PROMPT, RETRY_PROMPT):
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        image_part,
                    ],
                }
            ],
            max_tokens=150,
        )
        text = _clean_vision_text(response.choices[0].message.content)
        if text and not is_garbage_vision_response(text):
            return text

    return text or "Worker performing an assembly or inspection task."


def _normalize_for_compare(text: str) -> str:
    text = html.unescape(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def dedupe_similar_descriptions(descriptions: list, similarity_threshold: float = 0.52) -> list:
    """Collapse near-duplicate frame notes (wording differs, action is the same)."""
    unique = []
    for desc in descriptions:
        desc = html.unescape(desc.strip())
        if not desc:
            continue
        norm = _normalize_for_compare(desc)
        if any(
            SequenceMatcher(None, norm, _normalize_for_compare(existing)).ratio()
            >= similarity_threshold
            for existing in unique
        ):
            continue
        unique.append(desc)
    return unique


def summarize_visual_window(
    descriptions: list,
    start_label: str,
    end_label: str,
) -> str:
    """Turn many per-frame notes into one non-repetitive paragraph for a time window."""
    descriptions = [html.unescape(d.strip()) for d in descriptions if d and d.strip()]
    if not descriptions:
        return "(No visual notes for this segment)"

    unique = dedupe_similar_descriptions(descriptions)
    if len(unique) == 1:
        return unique[0]

    client = get_client()
    model = get_gpt4o_model()
    notes = "\n".join(f"- {d}" for d in unique[:12])
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You summarize factory video frame observations into clear prose. "
                    "Merge duplicate points; never repeat the same action in different words."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Time range: {start_label} to {end_label}\n\n"
                    "Below are per-frame notes from this period. Write ONE paragraph "
                    "(2-4 sentences) describing what the worker does. If the same inspection "
                    "or motion appears in multiple notes, mention it only once.\n\n"
                    f"{notes}"
                ),
            },
        ],
        max_tokens=220,
    )
    return _clean_vision_text(response.choices[0].message.content) or unique[0]
