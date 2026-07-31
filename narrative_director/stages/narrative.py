"""Stage 3a — Narrative understanding.

The LLM labels conversation events (questions, stories, emotional beats,
laughter, topic changes) over transcript windows. The mandatory
Off-Camera Rule is handled deterministically with a regex over word
timestamps — a production-control phrase is too important to leave to
model judgment.
"""

from __future__ import annotations

import re

from ..config import Config
from ..llm import LLMClient

STOP_PATTERNS = [
    r"\bstop\s+(?:the\s+)?rolling\b",
    r"\bstop\s+(?:the\s+)?roll\b",
    r"\bcut\s+the\s+cameras?\b",
]
RESTART_PATTERNS = [
    r"\brestart\s+(?:the\s+)?rolling\b",
    r"\bstart\s+(?:the\s+)?rolling\s+again\b",
    r"\bwe(?:'|\s+a)re\s+rolling\s+again\b",
]

NARRATIVE_PROMPT = """You are the story analyst for a podcast edit. Below are consecutive utterances with [start-end] times and the speaker role.

Identify narrative events. Types:
- "question"            host/guest asks something substantive
- "answer"              direct answer begins
- "story"               personal storytelling / anecdote (mark full span)
- "emotional"           vulnerable, moving, or intense moment (mark full span)
- "important_statement" key insight, thesis, quotable line
- "laughter"            audible/likely shared laughter
- "interruption"        one speaker cuts the other off
- "topic_change"        conversation moves to a new subject
- "banter"              rapid light back-and-forth

Rules:
- Use ONLY times inside [{w0} - {w1}].
- "intensity": 0.0-1.0 (how strongly the edit should honor this event).
- Be selective: report moments that should change CAMERA behavior, not every sentence.

Utterances:
{lines}

Reply ONLY with JSON:
{{"events": [{{"type": "...", "start": 0.0, "end": 0.0, "intensity": 0.5, "note": "<8 words max>"}}]}}"""


def find_off_camera_segments(words: list[dict], duration_s: float) -> list[dict]:
    """Deterministic scan for stop/restart-rolling directives (Rule 9)."""
    joined = ""
    index: list[tuple[int, dict]] = []  # (char offset, word)
    for w in words:
        index.append((len(joined), w))
        joined += w["w"].lower() + " "

    def word_at(char_pos: int) -> dict:
        best = words[0] if words else {"s": 0.0, "e": 0.0}
        for off, w in index:
            if off <= char_pos:
                best = w
            else:
                break
        return best

    marks: list[tuple[float, str]] = []
    for pat in STOP_PATTERNS:
        for m in re.finditer(pat, joined):
            marks.append((word_at(m.start())["s"], "stop"))
    for pat in RESTART_PATTERNS:
        for m in re.finditer(pat, joined):
            marks.append((word_at(m.end() - 1)["e"], "restart"))
    marks.sort()

    segments, open_at = [], None
    for t, kind in marks:
        if kind == "stop" and open_at is None:
            open_at = t
        elif kind == "restart" and open_at is not None:
            segments.append({"start": round(open_at, 3), "end": round(t, 3)})
            open_at = None
    if open_at is not None:  # never restarted — off-camera to the end
        segments.append({"start": round(open_at, 3), "end": round(duration_s, 3)})
    return segments


def find_silences(utterances: list[dict], duration_s: float, min_gap_s: float = 2.5) -> list[dict]:
    events = []
    prev_end = 0.0
    for u in utterances:
        if u["start"] - prev_end >= min_gap_s:
            events.append({"type": "silence", "start": prev_end, "end": u["start"], "intensity": 0.5, "note": "silence"})
        prev_end = max(prev_end, u["end"])
    if duration_s - prev_end >= min_gap_s:
        events.append({"type": "silence", "start": prev_end, "end": duration_s, "intensity": 0.5, "note": "silence"})
    return events


def analyze_narrative(
    cfg: Config, utterances: list[dict], words: list[dict], duration_s: float, llm: LLMClient
) -> dict:
    off_camera = find_off_camera_segments(words, duration_s)
    events = find_silences(utterances, duration_s)
    warnings: list[str] = []

    window: list[dict] = []
    win_chars = 0

    def flush():
        nonlocal window, win_chars
        if not window:
            return
        w0, w1 = window[0]["start"], window[-1]["end"]
        lines = "\n".join(
            f"[{u['start']:.1f}-{u['end']:.1f}] {u.get('speaker', '?')}: {u['text']}" for u in window
        )
        try:
            ans = llm.ask_json(NARRATIVE_PROMPT.format(w0=f"{w0:.1f}", w1=f"{w1:.1f}", lines=lines))
            for ev in ans.get("events", []):
                try:
                    s = max(w0, float(ev["start"]))
                    e = min(w1, float(ev["end"]))
                    if e <= s:
                        continue
                    events.append(
                        {
                            "type": str(ev.get("type", "other")),
                            "start": round(s, 3),
                            "end": round(e, 3),
                            "intensity": max(0.0, min(1.0, float(ev.get("intensity", 0.5)))),
                            "note": str(ev.get("note", ""))[:80],
                        }
                    )
                except (KeyError, TypeError, ValueError):
                    continue
        except RuntimeError as e:
            warnings.append(f"Narrative window {w0:.0f}-{w1:.0f}s failed: {e}")
        window, win_chars = [], 0

    for u in utterances:
        window.append(u)
        win_chars += len(u["text"])
        if win_chars > 4000 or (window and u["end"] - window[0]["start"] > 120):
            flush()
    flush()

    events.sort(key=lambda e: e["start"])
    return {"events": events, "off_camera_segments": off_camera, "warnings": warnings}
