"""Stage 2b — Speaker mapping via vision-correlated active speaker detection.

Design decision (documented for the review): audio-only diarization can
label turns "Speaker A/B" but can never tie a voice to a camera tile.
Since we must measure per-tile lip/mouth activity anyway to do that
mapping, we use the visual signal itself as the diarizer:

  1. ONE sequential ffmpeg decode pass at low fps produces, per tile:
       - mouth-region motion series  (speaker detection)
       - full-tile motion series     (freeze / dead-feed detection, reused later)
  2. Each transcript utterance is assigned to the hero tile whose
     z-scored mouth activity is highest during that utterance.
  3. An LLM pass over the opening conversation decides which tile is
     the HOST (show intros, question-asking) vs the GUEST.

A HITL checkpoint lets the human confirm/override the result.
"""

from __future__ import annotations

import subprocess

import numpy as np

from ..config import Config
from ..llm import LLMClient
from ..media.probe import MediaInfo

HOST_PROMPT = """Below is the opening of a podcast transcript. Each line is tagged with the camera tile of the person speaking (or "unknown").

Decide which tile belongs to the HOST (welcomes viewers, introduces the show/guest, asks the questions) and which to the GUEST (is introduced, answers, tells stories).

Tiles that appear: {tiles}

Transcript:
{lines}

Reply ONLY with JSON:
{{"host_tile": "<cam_id or null>", "guest_tile": "<cam_id or null>", "confidence": 0.0, "reason": "<one sentence>"}}"""


# ---------------------------------------------------------- motion extraction
def compute_motion_series(
    video: str, info: MediaInfo, inventory: dict, fps: float, scale_w: int = 640
) -> dict:
    """Single decode pass -> per-tile mouth-region and full-tile motion series."""
    scale_h = round(info.height * scale_w / info.width / 2) * 2
    sx, sy = scale_w / info.width, scale_h / info.height

    regions = {}
    for t in inventory["tiles"]:
        x, y, w, h = t["rect"]
        tile = (int(x * sx), int(y * sy), max(2, int(w * sx)), max(2, int(h * sy)))
        if t.get("face_rect"):
            fx, fy, fw, fh = t["face_rect"]
            # mouth region = lower half of the face box, in scaled full-frame coords
            mouth = (
                int((x + fx) * sx),
                int((y + fy + fh * 0.55) * sy),
                max(2, int(fw * sx)),
                max(2, int(fh * 0.45 * sy)),
            )
        else:  # no face found: central third of the tile as a fallback
            mouth = (
                tile[0] + tile[2] // 3,
                tile[1] + tile[3] // 3,
                max(2, tile[2] // 3),
                max(2, tile[3] // 3),
            )
        regions[t["id"]] = {"tile": tile, "mouth": mouth}

    proc = subprocess.Popen(
        [
            "ffmpeg", "-v", "error", "-i", video,
            "-vf", f"fps={fps},scale={scale_w}:{scale_h}",
            "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1",
        ],
        stdout=subprocess.PIPE,
    )
    frame_bytes = scale_w * scale_h
    prev = None
    mouth_series: dict[str, list[float]] = {k: [] for k in regions}
    tile_series: dict[str, list[float]] = {k: [] for k in regions}
    assert proc.stdout is not None
    while True:
        buf = proc.stdout.read(frame_bytes)
        if len(buf) < frame_bytes:
            break
        frame = np.frombuffer(buf, dtype=np.uint8).reshape(scale_h, scale_w)
        if prev is not None:
            diff = np.abs(frame.astype(np.int16) - prev.astype(np.int16)).astype(np.uint8)
            for cam, reg in regions.items():
                mx, my, mw, mh = reg["mouth"]
                tx, ty, tw, th = reg["tile"]
                mouth_series[cam].append(float(diff[my : my + mh, mx : mx + mw].mean()))
                tile_series[cam].append(float(diff[ty : ty + th, tx : tx + tw].mean()))
        prev = frame
    proc.wait()

    return {
        "fps": fps,
        "mouth": {k: np.array(v, dtype=np.float32) for k, v in mouth_series.items()},
        "tile": {k: np.array(v, dtype=np.float32) for k, v in tile_series.items()},
    }


def _zscore(x: np.ndarray) -> np.ndarray:
    std = x.std()
    return (x - x.mean()) / (std + 1e-6)


# ------------------------------------------------------------ speaker mapping
def map_speakers(
    cfg: Config,
    utterances: list[dict],
    inventory: dict,
    motion: dict,
    llm: LLMClient,
) -> dict:
    fps = motion["fps"]
    hero_ids = [t["id"] for t in inventory["tiles"] if t["role"] == "HERO"]
    if not hero_ids:
        hero_ids = [t["id"] for t in inventory["tiles"] if not t.get("empty_heuristic")]
    z = {cam: _zscore(motion["mouth"][cam]) for cam in hero_ids}
    min_act = float(cfg.analysis.get("face_min_activity", 0.15))

    warnings = []
    for u in utterances:
        i0 = int(u["start"] * fps)
        i1 = max(i0 + 1, int(u["end"] * fps))
        scores = {}
        for cam in hero_ids:
            series = z[cam]
            if i0 >= len(series):
                continue
            scores[cam] = float(series[i0 : min(i1, len(series))].mean())
        if not scores:
            u["tile"], u["speaker_conf"] = "unknown", 0.0
            continue
        best = max(scores, key=scores.get)  # type: ignore[arg-type]
        ranked = sorted(scores.values(), reverse=True)
        margin = ranked[0] - ranked[1] if len(ranked) > 1 else ranked[0]
        if scores[best] < min_act:
            u["tile"] = "unknown"  # off-camera voice (producer, crew)
            u["speaker_conf"] = 0.0
        else:
            u["tile"] = best
            u["speaker_conf"] = round(min(1.0, max(0.0, margin)), 2)

    # smooth: single-utterance blips between two same-tile neighbors
    for i in range(1, len(utterances) - 1):
        a, b, c = utterances[i - 1], utterances[i], utterances[i + 1]
        if b["tile"] != a["tile"] and a["tile"] == c["tile"] and b["speaker_conf"] < 0.1 and b["end"] - b["start"] < 2.0:
            b["tile"] = a["tile"]
    unknown_ratio = sum(1 for u in utterances if u["tile"] == "unknown") / max(1, len(utterances))
    if unknown_ratio > 0.25:
        warnings.append(
            f"{unknown_ratio:.0%} of utterances could not be tied to a camera — "
            "speaker mapping needs human review"
        )

    # ---- host vs guest via LLM on the opening conversation
    opening = [u for u in utterances if u["tile"] != "unknown"][:40]
    lines = "\n".join(f"[{u['tile']}] {u['text']}" for u in opening)
    seen_tiles = sorted({u["tile"] for u in opening})
    host_tile = guest_tile = None
    reason, conf = "", 0.0
    if len(seen_tiles) >= 1 and lines.strip():
        ans = llm.ask_json(HOST_PROMPT.format(tiles=", ".join(seen_tiles), lines=lines[:6000]))
        host_tile, guest_tile = ans.get("host_tile"), ans.get("guest_tile")
        reason, conf = ans.get("reason", ""), float(ans.get("confidence", 0) or 0)

    # cross-check with the vision stage's likely_host guesses
    vision_host = next(
        (t["id"] for t in inventory["tiles"] if t.get("likely_host") is True and t["role"] == "HERO"),
        None,
    )
    if vision_host and host_tile and vision_host != host_tile:
        warnings.append(
            f"Vision guessed host={vision_host} but transcript analysis says host={host_tile} "
            "— using transcript; confirm at HITL checkpoint"
        )
    if not host_tile and vision_host:
        host_tile = vision_host
    if not guest_tile:
        guest_tile = next((c for c in hero_ids if c != host_tile), None)

    speaker_mapping = {
        "host": {"tile": host_tile, "camera_label": "CAM_HOST_HERO"},
        "guest": {"tile": guest_tile, "camera_label": "CAM_GUEST_HERO"},
        "confidence": conf,
        "reason": reason,
    }

    # finalize named assignments on the inventory
    assignments = dict(inventory.get("assignments", {}))
    if host_tile:
        assignments["CAM_HOST_HERO"] = host_tile
    if guest_tile:
        assignments["CAM_GUEST_HERO"] = guest_tile
    inventory["assignments"] = assignments

    # role per utterance
    for u in utterances:
        if u["tile"] == host_tile:
            u["speaker"] = "host"
        elif u["tile"] == guest_tile:
            u["speaker"] = "guest"
        elif u["tile"] == "unknown":
            u["speaker"] = "unknown"
        else:
            u["speaker"] = u["tile"]  # third participant on another hero cam

    return {"speaker_mapping": speaker_mapping, "utterances": utterances, "warnings": warnings}
