#!/usr/bin/env python3
"""End-to-end pipeline test on the synthetic SyncMaster video.

Uses a MockLLM with canned, schedule-consistent answers so every
deterministic stage (grid detection, motion analysis, transcription,
off-camera scan, decision engine, XML writers, validation) is exercised
without an API key. Run with a real key + real footage via run.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from narrative_director.config import load_config  # noqa: E402
from narrative_director.pipeline import Pipeline  # noqa: E402


class MockLLM:
    class usage:  # noqa: N801 — mimic LLMUsage shape
        calls = 0
        vision_calls = 0
        errors: list = []

    def ask_json(self, prompt: str, system: str = "", images_png=None) -> dict:
        MockLLM.usage.calls += 1
        if images_png:
            MockLLM.usage.vision_calls += 1
        if "contact sheet" in prompt:
            return {
                "tiles": [
                    {"id": "cam_1", "role": "HERO", "person_count": 1,
                     "person_desc": "host, dark blue background", "likely_host": True, "confidence": 0.9},
                    {"id": "cam_2", "role": "EMPTY", "person_count": 0, "person_desc": "", "likely_host": None, "confidence": 0.95},
                    {"id": "cam_3", "role": "HERO", "person_count": 1,
                     "person_desc": "guest, maroon background", "likely_host": False, "confidence": 0.9},
                    {"id": "cam_4", "role": "EMPTY", "person_count": 0, "person_desc": "", "likely_host": None, "confidence": 0.95},
                    {"id": "cam_5", "role": "WIDE", "person_count": 2, "person_desc": "both, green set", "likely_host": None, "confidence": 0.85},
                    {"id": "cam_6", "role": "EMPTY", "person_count": 0, "person_desc": "", "likely_host": None, "confidence": 0.95},
                ],
                "notes": "3x2 grid, three live feeds",
            }
        if "host_tile" in prompt:
            return {"host_tile": "cam_1", "guest_tile": "cam_3", "confidence": 0.9,
                    "reason": "cam_1 welcomes viewers and asks the questions"}
        if "story analyst" in prompt:
            return {"events": [
                {"type": "question", "start": 15.0, "end": 20.0, "intensity": 0.6, "note": "origin question"},
                {"type": "story", "start": 21.0, "end": 35.0, "intensity": 0.8, "note": "failure story"},
                {"type": "emotional", "start": 26.0, "end": 34.0, "intensity": 0.85, "note": "vulnerable moment"},
                {"type": "question", "start": 44.0, "end": 49.0, "intensity": 0.5, "note": "advice question"},
            ]}
        if "CURRENT SPEAKER" in prompt:
            return {"frames": [{"time": 51.5, "adjustment": "mic_adjust", "reaction": None, "confidence": 0.8}]}
        if "LISTENER" in prompt:
            return {"frames": [{"time": 28.0, "reaction": "moved", "confidence": 0.7}]}
        if "rows" in prompt:
            return {"rows": 2, "cols": 3, "notes": "fallback"}
        return {}


def main() -> int:
    video = ROOT / "test_media" / "syncmaster_test.mp4"
    if not video.exists():
        print("Run scripts/make_synthetic.py first")
        return 2
    cfg = load_config(ROOT / "config.yaml")
    cfg.transcription["model_size"] = "tiny"  # fast for tests
    cfg.output["dir"] = "test_media/output"
    cfg.output["cache_dir"] = "test_media/.cache"

    p = Pipeline(cfg, str(video), auto=True, force="--force" in sys.argv)
    p.llm = MockLLM()
    report = p.run()

    # ---- assertions -------------------------------------------------------
    fails: list[str] = []
    meta = report["metadata"]
    inv = report["camera_inventory"]

    if inv["grid"]["rows"] != 2 or inv["grid"]["cols"] != 3:
        fails.append(f"grid detection: expected 2x3, got {inv['grid']}")
    a = inv["assignments"]
    if a.get("CAM_HOST_HERO") != "cam_1" or a.get("CAM_GUEST_HERO") != "cam_3" or a.get("CAM_WIDE") != "cam_5":
        fails.append(f"assignments wrong: {a}")
    if not report["off_camera_segments"]:
        fails.append("Rule 9: 'stop rolling' segment not detected")
    else:
        seg = report["off_camera_segments"][0]
        if not (30 < seg["start"] < 45 and seg["end"] > seg["start"]):
            fails.append(f"off-camera segment timing suspicious: {seg}")
    marker_names = {m["name"] for c in report["cuts"] for m in c.get("markers", [])}
    if "OFF_CAMERA_BRAINSTORM" not in marker_names:
        fails.append("OFF_CAMERA_BRAINSTORM marker missing from cuts")
    if "PHY_ADJ_CUT" not in marker_names:
        fails.append("PHY_ADJ_CUT marker missing from cuts")
    if not meta["validation"]["passed"]:
        fails.append(f"validation failed: {meta['validation']['errors']}")
    if meta["wide_usage_ratio"] >= 0.20:
        fails.append(f"Nav Thethi wide budget exceeded: {meta['wide_usage_ratio']}")
    hosted = [c for c in report["cuts"] if c["cameras"] == ["cam_1"]]
    guested = [c for c in report["cuts"] if c["cameras"] == ["cam_3"]]
    if not hosted or not guested:
        fails.append("expected cuts on both hero cameras")

    print("\n===== E2E RESULT =====")
    print(f"cuts: {meta['total_shots']}  avg shot: {meta['avg_shot_s']}s  wide: {meta['wide_usage_ratio']:.0%}")
    print(f"markers: {sorted(marker_names)}")
    print(f"off_camera: {report['off_camera_segments']}")
    print(f"validation: {meta['validation']['checks']}")
    if fails:
        print("\nFAILURES:")
        for f in fails:
            print(f"  ✗ {f}")
        return 1
    print("\nALL CHECKS PASSED ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
