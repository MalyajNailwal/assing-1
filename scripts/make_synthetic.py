#!/usr/bin/env python3
"""Build a synthetic SyncMaster test video.

Layout: 1920x1080, 3x2 grid of 640x540 tiles
  cam_1 host hero (blinking box while host speaks)
  cam_2 empty (black)
  cam_3 guest hero (blinking box while guest speaks)
  cam_4 empty (black)
  cam_5 wide (both boxes, gentle constant motion)
  cam_6 empty (black)

Audio: macOS `say` two-voice dialogue on a known schedule, including a
"stop rolling ... restart rolling" span to exercise Rule 9.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "test_media"
DUR = 60.0

# (speaker, start, text) — speech rate ~180wpm keeps these inside their slots
SCRIPT = [
    ("host", 0.5, "Welcome back to the show everyone. Today I am joined by an amazing guest. How are you doing today?"),
    ("guest", 8.0, "I am doing great, thanks for having me. I have been really looking forward to this conversation for a long time."),
    ("host", 15.0, "Let us start with your story. What was the moment everything changed for you?"),
    ("guest", 21.0, "Honestly, it was when my first company failed. I remember sitting alone in the office at midnight, and I realized that everything I believed about success was wrong. That failure taught me more than any win ever did."),
    ("host", 36.0, "Wow. That is powerful. Stop rolling please, let us discuss something quickly."),
    ("host", 43.0, "Okay restart rolling. So tell me, what advice would you give to young founders?"),
    ("guest", 50.0, "Keep your burn rate low, and talk to your customers every single day. That is the whole secret."),
]

VOICES = {"host": "Daniel", "guest": "Samantha"}


def run(cmd: list[str]):
    subprocess.run(cmd, check=True)


def build_audio(tmp: Path) -> Path:
    segs = []
    for i, (spk, start, text) in enumerate(SCRIPT):
        f = tmp / f"seg_{i}.aiff"
        run(["say", "-v", VOICES[spk], "-r", "185", "-o", str(f), text])
        segs.append((start, f))
    # overlay each segment at its start time on a silent bed
    inputs, filters = [], []
    for i, (start, f) in enumerate(segs):
        inputs += ["-i", str(f)]
        filters.append(f"[{i}:a]adelay={int(start * 1000)}|{int(start * 1000)}[a{i}]")
    mix = "".join(f"[a{i}]" for i in range(len(segs)))
    filters.append(f"{mix}amix=inputs={len(segs)}:normalize=0[aout]")
    out = tmp / "dialogue.wav"
    run(["ffmpeg", "-y", "-v", "error", *inputs,
         "-filter_complex", ";".join(filters),
         "-map", "[aout]", "-t", str(DUR), "-ar", "48000", "-ac", "2", str(out)])
    return out


def speech_windows(speaker: str) -> list[tuple[float, float]]:
    wins = []
    for i, (spk, start, _) in enumerate(SCRIPT):
        if spk != speaker:
            continue
        end = SCRIPT[i + 1][1] - 0.5 if i + 1 < len(SCRIPT) else DUR - 4
        wins.append((start, end))
    return wins


def blink_expr(windows: list[tuple[float, float]]) -> str:
    """enable expression: blink at 2.5Hz inside the speaking windows."""
    spans = "+".join(f"between(t,{a},{b})" for a, b in windows)
    return f"({spans})*lt(mod(t,0.4),0.2)"


def main():
    OUT_DIR.mkdir(exist_ok=True)
    tmp = OUT_DIR / "tmp"
    tmp.mkdir(exist_ok=True)
    audio = build_audio(tmp)

    host_w = speech_windows("host")
    guest_w = speech_windows("guest")
    tw, th = 640, 540

    # tiles: colored backgrounds; "mouth" = blinking box in the central third
    def tile_src(color):
        return f"color=c={color}:s={tw}x{th}:r=30:d={DUR}"

    fc = (
        f"{tile_src('0x3a4a5a')}[bg1];"
        f"[bg1]drawbox=x=270:y=230:w=100:h=80:color=white:t=fill:enable='{blink_expr(host_w)}',"
        f"drawtext=text='HOST CAM':x=20:y=20:fontsize=36:fontcolor=white,"
        f"noise=alls=14:allf=t+u[t1];"
        f"{tile_src('black')}[t2];"
        f"{tile_src('0x5a3a4a')}[bg3];"
        f"[bg3]drawbox=x=270:y=230:w=100:h=80:color=white:t=fill:enable='{blink_expr(guest_w)}',"
        f"drawtext=text='GUEST CAM':x=20:y=20:fontsize=36:fontcolor=white,"
        f"noise=alls=14:allf=t+u[t3];"
        f"{tile_src('black')}[t4];"
        f"{tile_src('0x2a4a3a')}[bg5];"
        f"[bg5]drawbox=x=100:y=200:w=60:h=50:color=gray:t=fill:enable='lt(mod(t,1.0),0.5)',"
        f"drawbox=x=480:y=200:w=60:h=50:color=gray:t=fill:enable='gte(mod(t,1.0),0.5)',"
        f"drawtext=text='WIDE':x=20:y=20:fontsize=36:fontcolor=white,"
        f"noise=alls=14:allf=t+u[t5];"
        f"{tile_src('black')}[t6];"
        f"[t1][t2][t3]hstack=3[row1];[t4][t5][t6]hstack=3[row2];"
        f"[row1][row2]vstack=2[vout]"
    )

    out = OUT_DIR / "syncmaster_test.mp4"
    run(["ffmpeg", "-y", "-v", "error",
         "-i", str(audio),
         "-filter_complex", fc,
         "-map", "[vout]", "-map", "0:a",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
         "-c:a", "aac", "-t", str(DUR), str(out)])
    (OUT_DIR / "show_type.txt").write_text("The Nav Thethi Show\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
