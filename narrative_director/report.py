"""editing_report.json — the pipeline's inspectable reasoning trail."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .media.probe import MediaInfo


def build_report(
    info: MediaInfo,
    show_type: str,
    inventory: dict,
    speaker_mapping: dict,
    cuts: list[dict],
    off_camera_segments: list[dict],
    validation: dict,
    warnings: list[str],
    hitl_overrides: list[dict],
    llm_usage: dict,
) -> dict:
    wide = inventory["assignments"].get("CAM_WIDE")
    total = cuts[-1]["end"] if cuts else 0.0
    # wide usage is measured against PUBLISHABLE runtime: off-camera segments
    # are marked for removal, so they count toward neither side of the ratio
    off_time = sum(c["duration"] for c in cuts if c["rule"] == "OFF_CAMERA_BRAINSTORM")
    total = max(0.001, total - off_time)
    wide_time = sum(
        c["duration"]
        for c in cuts
        if c["kind"] == "single" and c["cameras"] == [wide] and c["rule"] != "OFF_CAMERA_BRAINSTORM"
    ) if wide else 0.0
    return {
        "camera_inventory": {
            "grid": inventory["grid"],
            "assignments": inventory["assignments"],
            "tiles": [
                {k: t.get(k) for k in ("id", "rect", "role", "person_desc", "confidence", "face_rect")}
                for t in inventory["tiles"]
            ],
        },
        "speaker_mapping": speaker_mapping,
        "cuts": cuts,
        "warnings": sorted(set(warnings)),
        "off_camera_segments": off_camera_segments,
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": info.path,
            "duration_s": info.duration_s,
            "fps": info.fps,
            "show_type": show_type,
            "total_shots": len(cuts),
            "avg_shot_s": round(total / len(cuts), 2) if cuts else 0,
            "wide_usage_ratio": round(wide_time / total, 3) if total else 0,
            "validation": validation,
            "hitl_overrides": hitl_overrides,
            "llm_usage": llm_usage,
        },
    }


def write_report(report: dict, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(report, indent=2))
    return path
