"""Stage 2a — Transcription with word-level timestamps (faster-whisper).

Word timestamps are load-bearing: the Safety Rule (never cut mid-word)
and the Off-Camera Rule ("stop rolling") both depend on them.
"""

from __future__ import annotations

from pathlib import Path

from ..config import Config


def transcribe(cfg: Config, wav_path: str | Path) -> dict:
    from faster_whisper import WhisperModel  # deferred: heavy import

    tc = cfg.transcription
    device = tc.get("device", "auto")
    compute = tc.get("compute_type", "auto")
    if device == "auto":
        device = "cpu"  # CTranslate2 has no MPS backend; CPU is reliable on macOS
    if compute == "auto":
        compute = "int8" if device == "cpu" else "float16"

    model = WhisperModel(tc.get("model_size", "small"), device=device, compute_type=compute)
    segments, seg_info = model.transcribe(
        str(wav_path),
        language=tc.get("language") or None,
        word_timestamps=True,
        vad_filter=True,
    )

    out_segments = []
    for seg in segments:
        words = [
            {"w": w.word.strip(), "s": round(w.start, 3), "e": round(w.end, 3)}
            for w in (seg.words or [])
            if w.word.strip()
        ]
        if not words:
            continue
        out_segments.append(
            {
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "text": seg.text.strip(),
                "words": words,
            }
        )

    return {"language": seg_info.language, "segments": out_segments}


def build_utterances(transcript: dict, max_gap_s: float = 1.0) -> list[dict]:
    """Merge whisper segments into speaker turns (utterances) split on silence gaps."""
    utterances: list[dict] = []
    for seg in transcript["segments"]:
        if utterances and seg["start"] - utterances[-1]["end"] <= max_gap_s:
            u = utterances[-1]
            u["end"] = seg["end"]
            u["text"] = (u["text"] + " " + seg["text"]).strip()
            u["words"].extend(seg["words"])
        else:
            utterances.append(
                {
                    "id": len(utterances),
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"],
                    "words": list(seg["words"]),
                }
            )
    for i, u in enumerate(utterances):
        u["id"] = i
    return utterances


def all_words(transcript: dict) -> list[dict]:
    words = [w for seg in transcript["segments"] for w in seg["words"]]
    return sorted(words, key=lambda w: w["s"])
