"""Frame/audio extraction helpers built on ffmpeg + OpenCV."""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np


def extract_audio_wav(video: str, out_wav: str | Path, sample_rate: int = 16000) -> Path:
    """Extract mono 16k WAV for transcription/VAD."""
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error", "-i", video,
            "-vn", "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le",
            str(out_wav),
        ],
        check=True,
    )
    return out_wav


def sample_frames(video: str, times_s: list[float]) -> list[np.ndarray]:
    """Grab BGR frames at the given timestamps (seconds)."""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open {video}")
    frames = []
    for t in times_s:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000.0)
        ok, frame = cap.read()
        frames.append(frame if ok else None)
    cap.release()
    return frames


def iter_frames(video: str, start_s: float, end_s: float, fps: float):
    """Yield (t, frame) between start and end at ~fps, sequential decode (fast)."""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open {video}")
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, start_s) * 1000.0)
    step = 1.0 / fps
    next_t = start_s
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if t > end_s:
                break
            if t + 1e-6 >= next_t:
                yield t, frame
                next_t += step
    finally:
        cap.release()


def crop_tile(frame: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = rect
    return frame[y : y + h, x : x + w]


def resize_width(img: np.ndarray, width: int) -> np.ndarray:
    if img.shape[1] <= width:
        return img
    scale = width / img.shape[1]
    return cv2.resize(img, (width, max(1, int(img.shape[0] * scale))))


def png_bytes(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return buf.tobytes()


def contact_sheet(tiles: list[np.ndarray], labels: list[str], cols: int = 3, cell_w: int = 480) -> np.ndarray:
    """Compose labeled tile crops into one annotated sheet for the vision LLM."""
    cells = []
    for img, label in zip(tiles, labels):
        img = resize_width(img, cell_w)
        h, w = img.shape[:2]
        canvas = np.zeros((h + 36, cell_w, 3), dtype=np.uint8)
        canvas[36 : 36 + h, :w] = img
        cv2.putText(canvas, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cells.append(canvas)
    rows = []
    max_h = max(c.shape[0] for c in cells)
    cells = [cv2.copyMakeBorder(c, 0, max_h - c.shape[0], 0, 0, cv2.BORDER_CONSTANT) for c in cells]
    for i in range(0, len(cells), cols):
        row = cells[i : i + cols]
        while len(row) < cols:
            row.append(np.zeros_like(cells[0]))
        rows.append(np.hstack(row))
    return np.vstack(rows)
