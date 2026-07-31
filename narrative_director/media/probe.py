"""ffprobe wrapper — source media metadata."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


@dataclass
class MediaInfo:
    path: str
    duration_s: float
    width: int
    height: int
    fps: float
    fps_num: int
    fps_den: int
    has_audio: bool
    audio_sample_rate: int
    nb_frames: int

    @property
    def is_ntsc(self) -> bool:
        return self.fps_den == 1001


def probe(path: str | Path) -> MediaInfo:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-print_format", "json",
            "-show_format", "-show_streams", str(path),
        ],
        capture_output=True, text=True, check=True,
    ).stdout
    data = json.loads(out)
    video = next((s for s in data["streams"] if s["codec_type"] == "video"), None)
    if video is None:
        raise ValueError(f"No video stream in {path}")
    audio = next((s for s in data["streams"] if s["codec_type"] == "audio"), None)

    rate = Fraction(video.get("avg_frame_rate") or video.get("r_frame_rate") or "30/1")
    if rate == 0:
        rate = Fraction(video.get("r_frame_rate", "30/1"))
    duration = float(data["format"].get("duration") or video.get("duration") or 0.0)
    nb_frames = int(video.get("nb_frames") or round(duration * float(rate)))

    return MediaInfo(
        path=str(path.resolve()),
        duration_s=duration,
        width=int(video["width"]),
        height=int(video["height"]),
        fps=float(rate),
        fps_num=rate.numerator,
        fps_den=rate.denominator,
        has_audio=audio is not None,
        audio_sample_rate=int(audio["sample_rate"]) if audio else 0,
        nb_frames=nb_frames,
    )
