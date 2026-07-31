#!/usr/bin/env python3
"""AI Narrative Video Director — CLI entry point.

Usage:
  python run.py <syncmaster.mp4>            # full run with HITL checkpoints
  python run.py <syncmaster.mp4> --auto     # unattended (no checkpoints)
  python run.py <syncmaster.mp4> --force    # ignore stage caches, re-run all
  python run.py                             # auto-detect a single video in CWD
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".mxf", ".avi", ".m4v"}


def find_video(root: Path) -> str | None:
    vids = [p for p in root.iterdir() if p.suffix.lower() in VIDEO_EXTS and p.is_file()]
    if len(vids) == 1:
        return str(vids[0])
    if len(vids) > 1:
        print(f"Multiple videos found: {[v.name for v in vids]} — pass one explicitly.")
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="AI Narrative Video Director")
    ap.add_argument("video", nargs="?", help="SyncMaster video file")
    ap.add_argument("--config", default="config.yaml", help="config file path")
    ap.add_argument("--auto", action="store_true", help="skip human-in-the-loop checkpoints")
    ap.add_argument("--force", action="store_true", help="ignore stage caches")
    args = ap.parse_args()

    from narrative_director.config import load_config
    from narrative_director.pipeline import Pipeline

    cfg = load_config(args.config)
    video = args.video or find_video(cfg.root)
    if not video:
        print("No video specified and none found in the project folder.")
        return 2
    try:
        report = Pipeline(cfg, video, auto=args.auto, force=args.force).run()
    except SystemExit as e:
        print(str(e))
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted — stage caches are kept; re-run to resume.")
        return 130
    ok = report["metadata"]["validation"]["passed"]
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
