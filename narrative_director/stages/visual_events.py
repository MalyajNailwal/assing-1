"""Stage 3b — Visual event detection.

Three signal families:
  * Camera freezes / dead feeds  — deterministic, from the cached motion series
  * Physical adjustments (Rule 7) — vision LLM on active-speaker tile frames
  * Listener reactions   (Rule 2) — vision LLM on listener tile frames

All vision-LLM sampling shares one hard call budget (analysis.max_vision_calls)
so long episodes degrade gracefully instead of running up unbounded cost.
"""

from __future__ import annotations

import numpy as np

from ..config import Config
from ..llm import LLMClient
from ..media.extract import crop_tile, png_bytes, resize_width, sample_frames

PHYS_PROMPT = """These frames show the CURRENT SPEAKER on a podcast, camera tile "{cam}". Frame timestamps in seconds: {times}.

For each frame, report any physical adjustment (mandatory editorial rule — the edit must cut away from these):
- "mic_adjust"     touching/moving the microphone
- "face_scratch"   scratching/rubbing face
- "lip_lick"       licking lips
- "posture_change" clearly shifting sitting position
- "shoe_exposed"   foot/shoe visible in frame
- "device_gaze"    looking down at a tablet/monitor/phone (sustained)
Also report "reaction" if strongly smiling or laughing.

Reply ONLY with JSON:
{{"frames": [{{"time": <sec>, "adjustment": null_or_string, "reaction": null_or_string, "confidence": 0.0}}]}}"""

REACTION_PROMPT = """These frames show the LISTENER (not currently speaking) on a podcast, camera tile "{cam}". Frame timestamps in seconds: {times}.

For each frame report their visible reaction:
- "smile", "laugh", "nod", "surprise", "moved" (emotionally touched), or null if neutral/none.

Reply ONLY with JSON:
{{"frames": [{{"time": <sec>, "reaction": null_or_string, "confidence": 0.0}}]}}"""


# ------------------------------------------------------------- deterministic
def detect_freezes(motion: dict, inventory: dict, cfg: Config) -> list[dict]:
    """Tile static for >= freeze_min_s while other tiles move => frozen camera."""
    fps = motion["fps"]
    min_len = int(float(cfg.analysis.get("freeze_min_s", 1.0)) * fps)
    active = [t["id"] for t in inventory["tiles"] if t["role"] in ("HERO", "WIDE")]
    events = []
    for cam in active:
        series = motion["tile"][cam]
        frozen = series < 0.05
        i = 0
        while i < len(frozen):
            if frozen[i]:
                j = i
                while j < len(frozen) and frozen[j]:
                    j += 1
                if j - i >= min_len:
                    others = [motion["tile"][c][i:j].mean() for c in active if c != cam]
                    if others and float(np.mean(others)) > 0.2:  # scene alive, this cam dead
                        events.append(
                            {
                                "type": "camera_freeze",
                                "camera": cam,
                                "start": round(i / fps, 3),
                                "end": round(j / fps, 3),
                            }
                        )
                i = j
            else:
                i += 1
    return events


# ------------------------------------------------------------- vision budget
class VisionBudget:
    def __init__(self, max_calls: int):
        self.max_calls = max_calls
        self.used = 0
        self.exhausted = False

    def take(self) -> bool:
        if self.used >= self.max_calls:
            self.exhausted = True
            return False
        self.used += 1
        return True


def _batch_frames(video: str, cam_rect, times: list[float], tile_w: int) -> tuple[list[bytes], list[float]]:
    frames = sample_frames(video, times)
    imgs, kept = [], []
    for t, f in zip(times, frames):
        if f is None:
            continue
        imgs.append(png_bytes(resize_width(crop_tile(f, tuple(cam_rect)), tile_w)))
        kept.append(round(t, 2))
    return imgs, kept


def detect_visual_events(
    cfg: Config,
    video: str,
    inventory: dict,
    utterances: list[dict],
    motion: dict,
    llm: LLMClient,
) -> dict:
    interval = float(cfg.analysis.get("vision_interval_s", 8.0))
    tile_w = int(cfg.analysis.get("vision_tile_width", 512))
    budget = VisionBudget(int(cfg.analysis.get("max_vision_calls", 150)))
    rects = {t["id"]: t["rect"] for t in inventory["tiles"]}
    assign = inventory.get("assignments", {})
    host_tile, guest_tile = assign.get("CAM_HOST_HERO"), assign.get("CAM_GUEST_HERO")

    phys_events: list[dict] = []
    reaction_events: list[dict] = []
    warnings: list[str] = []

    def run_batch(cam: str, times: list[float], prompt_tpl: str, out: list[dict], kind: str):
        for i in range(0, len(times), 4):
            chunk = times[i : i + 4]
            if not budget.take():
                return
            imgs, kept = _batch_frames(video, rects[cam], chunk, tile_w)
            if not imgs:
                continue
            try:
                ans = llm.ask_json(prompt_tpl.format(cam=cam, times=kept), images_png=imgs)
            except RuntimeError as e:
                warnings.append(f"Vision batch failed ({cam} @ {kept[0] if kept else '?'}s): {e}")
                continue
            for fr in ans.get("frames", []):
                try:
                    t = float(fr["time"])
                except (KeyError, TypeError, ValueError):
                    continue
                # reject hallucinated timestamps outside the sampled window
                if not (min(kept) - 2.0 <= t <= max(kept) + 2.0):
                    continue
                adj = fr.get("adjustment")
                rea = fr.get("reaction")
                conf = float(fr.get("confidence", 0) or 0)
                if kind == "phys" and adj and conf >= 0.5:
                    out.append({"type": "phys_adjustment", "camera": cam, "time": t, "detail": adj, "confidence": conf})
                if rea and conf >= 0.5:
                    reaction_events.append({"type": "reaction", "camera": cam, "time": t, "detail": rea, "confidence": conf})

    for u in utterances:
        if u.get("tile") in (None, "unknown"):
            continue
        cam = u["tile"]
        dur = u["end"] - u["start"]
        if dur < 3.0 or cam not in rects:
            continue
        # active speaker: physical adjustment scan
        times = list(np.arange(u["start"] + 1.0, u["end"], interval))
        if times:
            run_batch(cam, times, PHYS_PROMPT, phys_events, "phys")
        # listener: reaction scan during longer turns only (cost control)
        listener = guest_tile if cam == host_tile else host_tile if cam == guest_tile else None
        if listener and listener in rects and dur >= 10.0:
            ltimes = list(np.arange(u["start"] + 2.0, u["end"], max(interval, 10.0)))
            if ltimes:
                run_batch(listener, ltimes, REACTION_PROMPT, reaction_events, "reaction")
        if budget.exhausted:
            break

    if budget.exhausted:
        warnings.append(
            f"Vision call budget ({budget.max_calls}) exhausted — visual events after "
            f"the cutoff were not analyzed. Raise analysis.max_vision_calls to cover the full episode."
        )

    freezes = detect_freezes(motion, inventory, cfg)

    def dedupe(evts: list[dict]) -> list[dict]:
        out: list[dict] = []
        for e in sorted(evts, key=lambda e: e["time"]):
            if out and out[-1]["camera"] == e["camera"] and out[-1]["detail"] == e["detail"] and e["time"] - out[-1]["time"] < 3.0:
                continue
            out.append(e)
        return out

    return {
        "phys_adjustments": dedupe(phys_events),
        "reactions": dedupe(reaction_events),
        "freezes": freezes,
        "vision_calls_used": budget.used,
        "warnings": warnings,
    }
