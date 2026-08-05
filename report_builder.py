import html
import json
import math
import os
import re
import shutil

from time_utils import seconds_to_timestamp, seconds_to_vtt
from vision_analysis import summarize_visual_window

_TIMESTAMP_RE = re.compile(r"\[([^\]]+?)\s*-\s*([^\]]+?)\]")


def _parse_time_token(token: str, video_duration_sec: float = 0.0) -> float:
    """Parse M:SS.ss labels or raw second values like 8.20 or 75.80."""
    token = (token or "").strip()
    if ":" in token:
        return timestamp_label_to_seconds(token, video_duration_sec)
    try:
        return float(token)
    except ValueError:
        return 0.0


def timestamp_label_to_seconds(label: str, video_duration_sec: float = 0.0) -> float:
    """Parse M:SS or H:MM:SS timestamp labels from work instructions."""
    label = (label or "").strip()
    parts = label.split(":")
    try:
        if len(parts) == 2:
            sec = int(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 3:
            sec = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        else:
            return 0.0
    except ValueError:
        return 0.0

    # GPT sometimes writes [43:00] meaning 43 seconds, not 43 minutes
    if video_duration_sec > 0 and sec > video_duration_sec * 1.05:
        if len(parts) == 2 and float(parts[1]) == 0.0:
            return float(parts[0])
    return sec


def _try_parse_step_line(body: str, video_duration_sec: float = 0.0) -> dict | None:
    """Extract a clickable step from a line containing [start - end]."""
    ts_match = _TIMESTAMP_RE.search(body)
    if not ts_match:
        return None
    start_sec = _parse_time_token(ts_match.group(1), video_duration_sec)
    end_sec = _parse_time_token(ts_match.group(2), video_duration_sec)
    step_text = re.sub(r"\*\*\[[^\]]+\]\*\*\s*", "", body)
    step_text = _TIMESTAMP_RE.sub("", step_text)
    step_text = _clean_md_inline(step_text).strip()
    return {
        "type": "step",
        "text": step_text,
        "start_sec": round(start_sec, 3),
        "end_sec": round(end_sec, 3),
        "start_label": seconds_to_timestamp(start_sec),
        "end_label": seconds_to_timestamp(end_sec),
        "subitems": [],
    }


def _clean_md_inline(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    return text.strip()


def parse_work_instructions_blocks(text: str, video_duration_sec: float = 0.0) -> list:
    """Parse markdown work instructions into structured blocks for the HTML report."""
    if not text or not text.strip():
        return []

    blocks = []
    for raw in text.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            continue

        if stripped == "---":
            blocks.append({"type": "divider"})
            continue

        if stripped.startswith("# ") and not stripped.startswith("##"):
            blocks.append({"type": "title", "text": _clean_md_inline(stripped[2:])})
            continue

        if stripped.startswith("### "):
            step = _try_parse_step_line(stripped[4:], video_duration_sec)
            if step:
                blocks.append(step)
            else:
                blocks.append({"type": "h3", "text": _clean_md_inline(stripped[4:])})
            continue

        if stripped.startswith("## "):
            blocks.append({"type": "h2", "text": _clean_md_inline(stripped[3:])})
            continue

        numbered = re.match(r"^(\d+)\.\s+(.*)", stripped)
        if numbered:
            body = numbered.group(2)
            step = _try_parse_step_line(body, video_duration_sec)
            if step:
                blocks.append(step)
                continue
            item_text = _clean_md_inline(stripped)
            if blocks and blocks[-1]["type"] == "step":
                blocks[-1].setdefault("subitems", []).append(item_text)
                continue
            blocks.append({"type": "numbered", "text": item_text})
            continue

        if raw[:1].isspace() and stripped.startswith(("- ", "* ")):
            bullet = _clean_md_inline(stripped[2:])
            step = _try_parse_step_line(bullet, video_duration_sec)
            if step:
                if blocks and blocks[-1]["type"] == "step":
                    blocks[-1].setdefault("subitems", []).append(step["text"])
                else:
                    blocks.append(step)
            elif blocks and blocks[-1]["type"] == "step":
                blocks[-1].setdefault("subitems", []).append(bullet)
            else:
                blocks.append({"type": "bullet", "text": bullet})
            continue

        if stripped.startswith(("- ", "* ")):
            body = stripped[2:]
            step = _try_parse_step_line(body, video_duration_sec)
            if step:
                blocks.append(step)
            else:
                blocks.append({"type": "bullet", "text": _clean_md_inline(body)})
            continue

        step = _try_parse_step_line(stripped, video_duration_sec)
        if step:
            blocks.append(step)
            continue

        para_text = _clean_md_inline(stripped)
        if blocks and blocks[-1].get("type") == "step":
            last = blocks[-1]
            last_text = (last.get("text") or "").strip()
            if not last_text:
                last["text"] = para_text
                continue
        blocks.append({"type": "paragraph", "text": para_text})

    return blocks


def _frame_index(filename: str) -> int:
    match = re.search(r"frame_(\d+)", filename)
    return int(match.group(1)) if match else 0


def enrich_visual_sequence(sequence: list, interval_seconds: float) -> list:
    enriched = []
    for step in sequence:
        start_idx = _frame_index(step["start"])
        end_idx = _frame_index(step["end"])
        enriched.append(
            {
                **step,
                "start_sec": round(start_idx * interval_seconds, 3),
                "end_sec": round((end_idx + 1) * interval_seconds, 3),
                "start_label": seconds_to_timestamp(start_idx * interval_seconds),
                "end_label": seconds_to_timestamp((end_idx + 1) * interval_seconds),
            }
        )
    return enriched


def enrich_transcript_segments(segments: list) -> list:
    enriched = []
    for i, segment in enumerate(segments):
        if segment.get("start_sec") is not None and segment.get("end_sec") is not None:
            start = float(segment["start_sec"])
            end = float(segment["end_sec"])
        else:
            start = float(segment.get("start", 0) or 0)
            end = segment.get("end")
            if end is None or float(end) <= start:
                end = start + 3.0
            end = float(end)

        enriched.append(
            {
                "id": segment.get("id", i),
                "text": segment.get("text", "").strip(),
                "start_sec": round(start, 3),
                "end_sec": round(end, 3),
                "start_label": seconds_to_timestamp(start),
                "end_label": seconds_to_timestamp(end),
            }
        )
    return enriched


def _overlaps(chunk_start: float, chunk_end: float, item_start: float, item_end: float) -> bool:
    return item_start < chunk_end and item_end > chunk_start


def _segments_need_redistribution(segments: list, video_duration_sec: float) -> bool:
    """True when Whisper returned one blob or timestamps don't cover the video."""
    if not segments:
        return True
    if len(segments) == 1:
        end = segments[0].get("end_sec", 0)
        if end < video_duration_sec * 0.25:
            return True
    # Many segments sharing the same start = double-enrichment or bad timestamps
    starts = [seg.get("start_sec", 0) for seg in segments]
    if len(segments) >= 3 and len(set(starts)) <= max(1, len(segments) // 3):
        return True
    covered = 0.0
    for seg in segments:
        start = max(0.0, seg.get("start_sec", 0))
        end = min(video_duration_sec, seg.get("end_sec", 0))
        covered += max(0.0, end - start)
    return covered < video_duration_sec * 0.3


def distribute_text_to_chunks(
    full_text: str,
    video_duration_sec: float,
    chunk_seconds: float,
) -> list:
    """Split full transcript text evenly across fixed time windows."""
    full_text = full_text.strip()
    if not full_text:
        return []

    num_chunks = max(1, math.ceil(video_duration_sec / chunk_seconds))
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", full_text) if s.strip()]
    if not sentences:
        sentences = [full_text]

    chunks = []
    sent_idx = 0
    total_sents = len(sentences)

    for i in range(num_chunks):
        start = i * chunk_seconds
        end = min((i + 1) * chunk_seconds, video_duration_sec)
        end_sent = int((i + 1) * total_sents / num_chunks)
        if i == num_chunks - 1:
            end_sent = total_sents
        part = " ".join(sentences[sent_idx:end_sent]).strip()
        sent_idx = end_sent
        chunks.append(
            {
                "id": i,
                "chunk_index": i,
                "start_sec": round(start, 3),
                "end_sec": round(end, 3),
                "start_label": seconds_to_timestamp(start),
                "end_label": seconds_to_timestamp(end),
                "text": part or "(No speech in this segment)",
            }
        )
    return chunks


def build_speech_chunks(
    transcript_segments: list,
    full_text: str,
    video_duration_sec: float,
    chunk_seconds: float = 10.0,
) -> list:
    """Build 10s transcript chunks — speech only, spread across full video."""
    already_enriched = bool(
        transcript_segments
        and transcript_segments[0].get("start_sec") is not None
    )
    enriched = transcript_segments if already_enriched else enrich_transcript_segments(transcript_segments)

    if _segments_need_redistribution(enriched, video_duration_sec):
        text = full_text or (enriched[0]["text"] if enriched else "")
        print(
            "Whisper timestamps incomplete — distributing transcript evenly "
            f"across {video_duration_sec:.0f}s in {chunk_seconds:.0f}s segments..."
        )
        return distribute_text_to_chunks(text, video_duration_sec, chunk_seconds)

    chunks = []
    chunk_index = 0
    window_start = 0.0

    while window_start < video_duration_sec:
        window_end = min(window_start + chunk_seconds, video_duration_sec)
        speech_parts = []
        for seg in enriched:
            if seg.get("text") and _overlaps(
                window_start, window_end, seg["start_sec"], seg["end_sec"]
            ):
                speech_parts.append(seg["text"])

        chunks.append(
            {
                "id": chunk_index,
                "chunk_index": chunk_index,
                "start_sec": round(window_start, 3),
                "end_sec": round(window_end, 3),
                "start_label": seconds_to_timestamp(window_start),
                "end_label": seconds_to_timestamp(window_end),
                "text": " ".join(speech_parts).strip() or "(No speech in this segment)",
            }
        )
        window_start += chunk_seconds
        chunk_index += 1

    return chunks


def prepare_visual_report_items(visual_enriched: list) -> list:
    """Format consolidated visual steps for the report (one row per action)."""
    items = []
    for i, step in enumerate(visual_enriched):
        action = html.unescape((step.get("action") or "").strip())
        items.append(
            {
                "id": i,
                "start_sec": step["start_sec"],
                "end_sec": step["end_sec"],
                "start_label": step["start_label"],
                "end_label": step["end_label"],
                "text": action,
                "action": action,
            }
        )
    return items


def build_visual_time_chunks(
    visual_sequence: list,
    video_duration_sec: float,
    chunk_seconds: float = 20.0,
    *,
    summarize: bool = True,
) -> list:
    """Group visual steps into fixed windows with one consolidated description each."""
    if video_duration_sec <= 0:
        video_duration_sec = chunk_seconds

    chunks = []
    chunk_index = 0
    window_start = 0.0
    total_windows = max(1, math.ceil(video_duration_sec / chunk_seconds))

    while window_start < video_duration_sec:
        window_end = min(window_start + chunk_seconds, video_duration_sec)
        start_label = seconds_to_timestamp(window_start)
        end_label = seconds_to_timestamp(window_end)
        visual_parts = []

        for step in visual_sequence:
            action = html.unescape((step.get("action") or step.get("text") or "").strip())
            if not action:
                continue
            start = step.get("start_sec", 0)
            end = step.get("end_sec", start + 1)
            if _overlaps(window_start, window_end, start, end):
                visual_parts.append(action)

        if summarize and visual_parts:
            print(
                f"  Summarizing visual segment {chunk_index + 1}/{total_windows} "
                f"({start_label} -> {end_label})..."
            )
            summary = summarize_visual_window(visual_parts, start_label, end_label)
        elif visual_parts:
            from vision_analysis import dedupe_similar_descriptions

            deduped = dedupe_similar_descriptions(visual_parts)
            summary = deduped[0] if len(deduped) == 1 else " ".join(deduped[:2])
        else:
            summary = "(No visual notes for this segment)"

        chunks.append(
            {
                "id": chunk_index,
                "chunk_index": chunk_index,
                "start_sec": round(window_start, 3),
                "end_sec": round(window_end, 3),
                "start_label": start_label,
                "end_label": end_label,
                "text": summary,
                "action": summary,
            }
        )
        window_start += chunk_seconds
        chunk_index += 1

    return chunks


def write_vtt(segments: list, path: str) -> None:
    lines = ["WEBVTT", ""]
    for i, seg in enumerate(segments):
        if not seg.get("text"):
            continue
        start = seconds_to_vtt(seg["start_sec"])
        end = seconds_to_vtt(seg["end_sec"])
        lines.append(f"{i + 1}")
        lines.append(f"{start} --> {end}")
        lines.append(seg["text"])
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_timed_transcript_txt(segments: list, path: str) -> None:
    lines = []
    for seg in segments:
        if not seg.get("text"):
            continue
        lines.append(f"[{seg['start_label']} - {seg['end_label']}] {seg['text']}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def copy_video_for_report(video_path: str, output_dir: str) -> str:
    ext = os.path.splitext(video_path)[1].lower() or ".mp4"
    dest_name = f"source_video{ext}"
    dest_path = os.path.join(output_dir, dest_name)
    if os.path.normcase(os.path.abspath(video_path)) == os.path.normcase(os.path.abspath(dest_path)):
        return dest_name
    shutil.copy2(video_path, dest_path)
    return dest_name


def build_interactive_report(
    output_dir: str,
    video_filename: str,
    transcript_chunks: list,
    transcript_segments: list,
    visual_sequence: list,
    work_instructions: str,
    video_title: str = "Video Work Instructions",
    chunk_seconds: float = 10.0,
    visual_chunk_seconds: float = 20.0,
    video_duration_sec: float = 0.0,
) -> str:
    report_path = os.path.join(output_dir, "report.html")
    wi_blocks = parse_work_instructions_blocks(work_instructions, video_duration_sec)
    payload = {
        "video": video_filename,
        "title": video_title,
        "chunkSeconds": chunk_seconds,
        "transcript": transcript_chunks,
        "transcriptRaw": transcript_segments,
        "visual": visual_sequence,
        "workInstructions": work_instructions,
        "workInstructionBlocks": wi_blocks,
    }

    data_json = json.dumps(payload, ensure_ascii=False)
    template = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>__TITLE__</title>
  <style>
    :root { --bg: #0f1419; --panel: #1a2332; --text: #e8eef7; --muted: #8b9cb3;
            --accent: #3b82f6; --accent-soft: #1e3a5f; --border: #2d3a4f; }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: "Segoe UI", system-ui, sans-serif; background: var(--bg); color: var(--text); }
    header { padding: 1rem 1.5rem; border-bottom: 1px solid var(--border); background: var(--panel); }
    h1 { margin: 0; font-size: 1.25rem; }
    .layout { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; padding: 1rem; min-height: calc(100vh - 64px); }
    @media (max-width: 900px) { .layout { grid-template-columns: 1fr; } }
    .video-wrap { background: #000; border-radius: 8px; overflow: hidden; position: sticky; top: 1rem; }
    video { width: 100%; max-height: 50vh; display: block; background: #000; }
    .time-display { padding: 0.5rem 1rem; background: var(--panel); font-family: monospace; color: var(--muted); }
    .panel { background: var(--panel); border: 1px solid var(--border); border-radius: 8px; display: flex; flex-direction: column; min-height: 400px; }
    .tabs { display: flex; border-bottom: 1px solid var(--border); }
    .tab { flex: 1; padding: 0.75rem; background: none; border: none; color: var(--muted); cursor: pointer; font-size: 0.9rem; }
    .tab.active { color: var(--accent); border-bottom: 2px solid var(--accent); background: var(--accent-soft); }
    .tab-content { display: none; padding: 0; overflow-y: auto; flex: 1; max-height: 70vh; }
    .tab-content.active { display: block; }
    .cue { padding: 0.75rem 1rem; border-bottom: 1px solid var(--border); cursor: pointer; transition: background 0.15s; }
    .cue:hover { background: var(--accent-soft); }
    .cue.active { background: var(--accent-soft); border-left: 3px solid var(--accent); }
    .cue-time { font-size: 0.75rem; color: var(--accent); font-family: monospace; margin-bottom: 0.25rem; }
    .cue-text { line-height: 1.45; white-space: pre-wrap; }
    .wi-wrap { padding: 1rem 1.25rem 1.5rem; }
    .wi-title { font-size: 1.15rem; font-weight: 600; margin: 0 0 1rem; color: var(--text); line-height: 1.35; }
    .wi-h2 { font-size: 0.95rem; font-weight: 600; margin: 1.25rem 0 0.5rem; color: var(--accent); text-transform: uppercase; letter-spacing: 0.04em; }
    .wi-h3 { font-size: 1rem; font-weight: 600; margin: 1rem 0 0.5rem; color: var(--text); }
    .wi-p { margin: 0.35rem 0 0.6rem; line-height: 1.55; color: var(--text); }
    .wi-divider { border: none; border-top: 1px solid var(--border); margin: 1rem 0; }
    .wi-bullet, .wi-numbered { margin: 0.25rem 0 0.25rem 1.1rem; line-height: 1.5; color: var(--muted); }
    .wi-step {
      margin: 0.5rem 0; padding: 0.75rem 0.9rem; border: 1px solid var(--border);
      border-radius: 8px; cursor: pointer; transition: background 0.15s, border-color 0.15s;
      background: rgba(15, 20, 25, 0.35);
    }
    .wi-step::before {
      content: "▶ Click to play";
      display: block;
      font-size: 0.65rem;
      color: var(--muted);
      margin-bottom: 0.35rem;
      letter-spacing: 0.03em;
      text-transform: uppercase;
    }
    .wi-step:hover { background: var(--accent-soft); border-color: var(--accent); }
    .wi-step.active { background: var(--accent-soft); border-color: var(--accent); border-left: 3px solid var(--accent); }
    .wi-step-time { font-size: 0.75rem; color: var(--accent); font-family: monospace; margin-bottom: 0.35rem; }
    .wi-step-text { line-height: 1.5; }
    .wi-sublist { margin: 0.5rem 0 0 1rem; padding: 0; list-style: disc; color: var(--muted); }
    .wi-sublist li { margin: 0.2rem 0; line-height: 1.45; }
    .empty { padding: 2rem; color: var(--muted); text-align: center; }
  </style>
</head>
<body>
  <header><h1 id="page-title">Video Work Instructions</h1></header>
  <div class="layout">
    <div>
      <div class="video-wrap">
        <video id="player" controls playsinline preload="metadata"></video>
        <div class="time-display" id="current-time">0:00.00</div>
      </div>
    </div>
    <div class="panel">
      <div class="tabs">
        <button class="tab active" data-tab="transcript">Transcript (__CHUNK_SEC__s)</button>
        <button class="tab" data-tab="visual">Visual steps (__VISUAL_CHUNK_SEC__s)</button>
        <button class="tab" data-tab="instructions">Work instructions</button>
      </div>
      <div id="transcript" class="tab-content active"></div>
      <div id="visual" class="tab-content"></div>
      <div id="instructions" class="tab-content"></div>
    </div>
  </div>
  <script>
    const DATA = __DATA_JSON__;
    const player = document.getElementById("player");
    const currentTimeEl = document.getElementById("current-time");
    document.getElementById("page-title").textContent = DATA.title;
    player.src = DATA.video;

    function formatTime(sec) {
      sec = Math.max(0, sec || 0);
      const h = Math.floor(sec / 3600);
      const m = Math.floor((sec % 3600) / 60);
      const s = sec % 60;
      const ss = s.toFixed(2).padStart(s < 10 ? 4 : 5, "0");
      return h > 0 ? `${h}:${String(m).padStart(2,"0")}:${ss}` : `${m}:${ss}`;
    }

    function seekTo(sec) {
      player.currentTime = Math.max(0, sec);
      player.play().catch(() => {});
    }

    function renderCues(container, items, type) {
      container.innerHTML = "";
      if (!items || !items.length) {
        container.innerHTML = '<div class="empty">No items available.</div>';
        return;
      }
      items.forEach((item, idx) => {
        const el = document.createElement("div");
        el.className = "cue";
        el.dataset.start = item.start_sec;
        el.dataset.end = item.end_sec;
        el.dataset.idx = idx;
        el.dataset.type = type;
        el.innerHTML = `<div class="cue-time">${item.start_label} → ${item.end_label}</div><div class="cue-text"></div>`;
        el.querySelector(".cue-text").textContent = item.text || item.action || "";
        el.addEventListener("click", () => seekTo(item.start_sec));
        container.appendChild(el);
      });
    }

    renderCues(document.getElementById("transcript"), DATA.transcript, "transcript");
    renderCues(document.getElementById("visual"), DATA.visual, "visual");
    renderWorkInstructions(document.getElementById("instructions"), DATA.workInstructionBlocks);

    function renderWorkInstructions(container, blocks) {
      container.innerHTML = "";
      if (!blocks || !blocks.length) {
        container.innerHTML = '<div class="empty">No work instructions generated.</div>';
        return;
      }
      const wrap = document.createElement("div");
      wrap.className = "wi-wrap";
      blocks.forEach((block, idx) => {
        if (block.type === "title") {
          const el = document.createElement("h2");
          el.className = "wi-title";
          el.textContent = block.text;
          wrap.appendChild(el);
        } else if (block.type === "h2") {
          const el = document.createElement("h3");
          el.className = "wi-h2";
          el.textContent = block.text;
          wrap.appendChild(el);
        } else if (block.type === "h3") {
          const el = document.createElement("h4");
          el.className = "wi-h3";
          el.textContent = block.text;
          wrap.appendChild(el);
        } else if (block.type === "paragraph") {
          const el = document.createElement("p");
          el.className = "wi-p";
          el.textContent = block.text;
          wrap.appendChild(el);
        } else if (block.type === "bullet") {
          const el = document.createElement("div");
          el.className = "wi-bullet";
          el.textContent = "• " + block.text;
          wrap.appendChild(el);
        } else if (block.type === "numbered") {
          const el = document.createElement("div");
          el.className = "wi-numbered";
          el.textContent = block.text;
          wrap.appendChild(el);
        } else if (block.type === "divider") {
          const el = document.createElement("hr");
          el.className = "wi-divider";
          wrap.appendChild(el);
        } else if (block.type === "step") {
          const el = document.createElement("div");
          el.className = "wi-step";
          el.dataset.start = block.start_sec;
          el.dataset.end = block.end_sec;
          el.dataset.idx = idx;
          el.dataset.type = "instruction";
          el.innerHTML =
            `<div class="wi-step-time">${block.start_label} → ${block.end_label}</div>` +
            `<div class="wi-step-text"></div>`;
          el.querySelector(".wi-step-text").textContent = block.text;
          if (block.subitems && block.subitems.length) {
            const ul = document.createElement("ul");
            ul.className = "wi-sublist";
            block.subitems.forEach(item => {
              const li = document.createElement("li");
              li.textContent = item;
              ul.appendChild(li);
            });
            el.appendChild(ul);
          }
          el.addEventListener("click", () => seekTo(block.start_sec));
          wrap.appendChild(el);
        }
      });
      container.appendChild(wrap);
    }

    document.querySelectorAll(".tab").forEach(btn => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".tab").forEach(b => b.classList.remove("active"));
        document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
        btn.classList.add("active");
        document.getElementById(btn.dataset.tab).classList.add("active");
      });
    });

    function highlightActive() {
      const t = player.currentTime;
      currentTimeEl.textContent = formatTime(t);
      document.querySelectorAll(".cue, .wi-step").forEach(el => {
        const start = parseFloat(el.dataset.start);
        const end = parseFloat(el.dataset.end);
        if (!Number.isNaN(start) && !Number.isNaN(end)) {
          el.classList.toggle("active", t >= start && t < end);
        }
      });
    }
    player.addEventListener("timeupdate", highlightActive);
  </script>
</body>
</html>"""

    html_content = (
        template.replace("__TITLE__", html.escape(video_title))
        .replace("__DATA_JSON__", data_json)
        .replace("__CHUNK_SEC__", str(int(chunk_seconds)))
        .replace("__VISUAL_CHUNK_SEC__", str(int(visual_chunk_seconds)))
    )
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    return report_path


def build_report(
    video_path: str,
    output_dir: str,
    transcript: dict,
    visual_sequence: list,
    work_instructions: str,
    frame_interval: float,
    video_duration_sec: float,
    chunk_seconds: float = 10.0,
    visual_chunk_seconds: float = 20.0,
    summarize_visual: bool = True,
    use_cached_visual_chunks: bool = False,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    video_filename = copy_video_for_report(video_path, output_dir)

    transcript_segments = enrich_transcript_segments(transcript.get("segments", []))
    visual_enriched = enrich_visual_sequence(visual_sequence, frame_interval)

    transcript_chunks = build_speech_chunks(
        transcript_segments,
        transcript.get("text", ""),
        video_duration_sec,
        chunk_seconds=chunk_seconds,
    )
    visual_chunks_path = os.path.join(output_dir, "visual_chunks.json")
    if use_cached_visual_chunks and os.path.isfile(visual_chunks_path):
        with open(visual_chunks_path, encoding="utf-8") as f:
            visual_chunks = json.load(f)
        print("Using cached visual segment summaries.")
    else:
        if summarize_visual:
            print("Consolidating visual steps into single descriptions per segment...")
        visual_chunks = build_visual_time_chunks(
            visual_enriched,
            video_duration_sec,
            chunk_seconds=visual_chunk_seconds,
            summarize=summarize_visual,
        )
        with open(visual_chunks_path, "w", encoding="utf-8") as f:
            json.dump(visual_chunks, f, indent=2)

    chunks_path = os.path.join(output_dir, "transcript_chunks.json")
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(transcript_chunks, f, indent=2)

    segments_path = os.path.join(output_dir, "transcript_segments.json")
    with open(segments_path, "w", encoding="utf-8") as f:
        json.dump(transcript_segments, f, indent=2)

    vtt_path = os.path.join(output_dir, "transcript.vtt")
    write_vtt(transcript_chunks, vtt_path)

    timed_txt_path = os.path.join(output_dir, "transcript_timed.txt")
    write_timed_transcript_txt(transcript_chunks, timed_txt_path)

    visual_path = os.path.join(output_dir, "visual_sequence.json")
    with open(visual_path, "w", encoding="utf-8") as f:
        json.dump(visual_enriched, f, indent=2)

    timeline = {
        "chunk_seconds": chunk_seconds,
        "visual_chunk_seconds": visual_chunk_seconds,
        "video_duration_sec": video_duration_sec,
        "transcript_chunks": transcript_chunks,
        "transcript_raw": transcript_segments,
        "visual_chunks": visual_chunks,
        "visual_steps": visual_enriched,
    }
    timeline_path = os.path.join(output_dir, "timeline.json")
    with open(timeline_path, "w", encoding="utf-8") as f:
        json.dump(timeline, f, indent=2)

    video_title = os.path.splitext(os.path.basename(video_path))[0]
    report_path = build_interactive_report(
        output_dir,
        video_filename,
        transcript_chunks,
        transcript_segments,
        visual_chunks,
        work_instructions,
        video_title=video_title,
        chunk_seconds=chunk_seconds,
        visual_chunk_seconds=visual_chunk_seconds,
        video_duration_sec=video_duration_sec,
    )

    return {
        "report_path": report_path,
        "vtt_path": vtt_path,
        "timed_transcript_path": timed_txt_path,
        "timeline_path": timeline_path,
        "chunks_path": chunks_path,
        "video_filename": video_filename,
    }
