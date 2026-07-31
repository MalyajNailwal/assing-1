"""Phase A of the assisted run (no API key): cameras with assistant-provided
role analysis, transcription, motion series, speaker mapping."""
import sys, json, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from narrative_director.config import load_config
from narrative_director.media.probe import probe
from narrative_director.media.extract import extract_audio_wav
from narrative_director.stages.camera_discovery import discover_cameras
from narrative_director.stages.transcribe import transcribe, build_utterances
from narrative_director.stages.speakers import compute_motion_series, map_speakers
import numpy as np

VIDEO = str(Path("Russ Finney Sync Master (10 FPS).mp4").resolve())

class AssistantLLM:
    """Claude's own analysis of this footage, injected in place of API calls."""
    def ask_json(self, prompt, system="", images_png=None):
        if "contact sheet" in prompt:
            return {"tiles": [
                {"id": "cam_1", "role": "HERO", "person_id": "A", "tightness": "close",
                 "person_count": 1, "person_desc": "older man, navy shirt (Russ) - profile close-up", "likely_host": False, "confidence": 0.9},
                {"id": "cam_2", "role": "HERO", "person_id": "B", "tightness": "medium",
                 "person_count": 1, "person_desc": "tattooed man, striped polo (Nav) - side angle", "likely_host": True, "confidence": 0.85},
                {"id": "cam_3", "role": "HERO", "person_id": "A", "tightness": "close",
                 "person_count": 1, "person_desc": "Russ - frontal medium close-up", "likely_host": False, "confidence": 0.95},
                {"id": "cam_4", "role": "HERO", "person_id": "B", "tightness": "medium",
                 "person_count": 1, "person_desc": "Nav - frontal medium", "likely_host": True, "confidence": 0.9},
                {"id": "cam_5", "role": "HERO", "person_id": "A", "tightness": "wide",
                 "person_count": 1, "person_desc": "Russ - wider medium showing hands", "likely_host": False, "confidence": 0.9},
            ], "notes": "freeform 2-column layout, 3 guest angles + 2 host angles, no two-person wide"}
        if "host_tile" in prompt:
            # provisional: Nav Thethi's frontal cam; verified against transcript in phase B
            return {"host_tile": "cam_4", "guest_tile": "cam_3", "confidence": 0.8,
                    "reason": "Nav Thethi (show host) on cam_4/cam_2; Russ Finney is the guest"}
        raise RuntimeError("unexpected LLM call in phase A: " + prompt[:80])

cfg = load_config("config.yaml")
llm = AssistantLLM()
cache = cfg.cache_dir / hashlib.md5(VIDEO.encode()).hexdigest()[:8]
cache.mkdir(parents=True, exist_ok=True)
info = probe(VIDEO)
print(f"probe: {info.width}x{info.height} @{info.fps} {info.duration_s:.0f}s")

inv = discover_cameras(cfg, info, llm)
(cache / "01_cameras.json").write_text(json.dumps(inv, indent=2))
print("cameras:", inv["assignments"], "| alts:", inv["alternates"])
print("layout:", inv["grid"], "| warnings:", inv["warnings"])

wav = cache / "audio.wav"
if not wav.exists():
    extract_audio_wav(VIDEO, wav)
print("transcribing (this is the slow part)...")
tr = transcribe(cfg, wav)
(cache / "02_transcript.json").write_text(json.dumps(tr, indent=2))
utts = build_utterances(tr)
print(f"transcript: {len(tr['segments'])} segments, {len(utts)} utterances, lang={tr['language']}")

motion = compute_motion_series(VIDEO, info, inv, fps=float(cfg.analysis.get("motion_fps", 4.0)))
np.savez_compressed(cache / "motion.npz", fps=np.array(motion["fps"]),
    **{f"mouth_{k}": v for k, v in motion["mouth"].items()},
    **{f"tile_{k}": v for k, v in motion["tile"].items()})
print("motion series done")

mapping = map_speakers(cfg, utts, inv, motion, llm)
(cache / "03_speakers.json").write_text(json.dumps(mapping, indent=2))
(cache / "01_cameras.json").write_text(json.dumps(inv, indent=2))  # assignments updated
sm = mapping["speaker_mapping"]
print("speakers:", sm, "| warnings:", mapping["warnings"])
known = [u for u in mapping["utterances"] if u["speaker"] != "unknown"]
print(f"utterances tied to a camera: {len(known)}/{len(mapping['utterances'])}")
