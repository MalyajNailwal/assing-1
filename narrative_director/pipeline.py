"""Pipeline orchestrator.

Runs the stages in order with per-stage JSON caching (.cache/): expensive
stages (transcription, vision) are never re-paid on a re-run unless
--force is given or the cache is deleted. HITL checkpoints sit between
stages; XML is only written after the human approves the cut list.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
from rich.console import Console

from .config import Config
from .hitl import HITL
from .llm import LLMClient
from .media.extract import extract_audio_wav
from .media.probe import MediaInfo, probe
from .report import build_report, write_report
from .stages import decision_engine as de
from .stages.camera_discovery import discover_cameras
from .stages.narrative import analyze_narrative
from .stages.speakers import compute_motion_series, map_speakers
from .stages.transcribe import all_words, build_utterances, transcribe
from .stages.validate import validate
from .stages.visual_events import detect_visual_events
from .stages.xml_fcp7 import write_fcp7_xml
from .stages.xml_fcpxml import write_fcpxml

console = Console()


class Pipeline:
    def __init__(self, cfg: Config, video: str, auto: bool = False, force: bool = False):
        self.cfg = cfg
        self.video = str(Path(video).resolve())
        self.force = force
        self.llm = LLMClient(cfg.llm)
        self.hitl = HITL(
            bool(cfg.hitl.get("enabled", True)),
            list(cfg.hitl.get("checkpoints", [])),
            auto=auto,
        )
        vid_hash = hashlib.md5(self.video.encode()).hexdigest()[:8]
        self.cache = cfg.cache_dir / vid_hash
        self.cache.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------- caching
    def _cached(self, name: str, fn):
        path = self.cache / f"{name}.json"
        if path.exists() and not self.force:
            console.print(f"[dim]stage {name}: cached ({path})[/dim]")
            return json.loads(path.read_text())
        console.print(f"[bold cyan]stage {name}: running[/bold cyan]")
        result = fn()
        path.write_text(json.dumps(result, indent=2))
        return result

    def _cached_motion(self, info: MediaInfo, inventory: dict) -> dict:
        path = self.cache / "motion.npz"
        if path.exists() and not self.force:
            data = np.load(path, allow_pickle=False)
            fps = float(data["fps"])
            cams = [t["id"] for t in inventory["tiles"]]
            return {
                "fps": fps,
                "mouth": {c: data[f"mouth_{c}"] for c in cams if f"mouth_{c}" in data},
                "tile": {c: data[f"tile_{c}"] for c in cams if f"tile_{c}" in data},
            }
        console.print("[bold cyan]stage motion: decoding video (single pass)[/bold cyan]")
        motion = compute_motion_series(
            self.video, info, inventory, fps=float(self.cfg.analysis.get("motion_fps", 4.0))
        )
        np.savez_compressed(
            path,
            fps=np.array(motion["fps"]),
            **{f"mouth_{k}": v for k, v in motion["mouth"].items()},
            **{f"tile_{k}": v for k, v in motion["tile"].items()},
        )
        return motion

    # ----------------------------------------------------------------- run
    def run(self) -> dict:
        cfg = self.cfg
        info = probe(self.video)
        console.print(
            f"[bold]Source[/bold]: {Path(self.video).name}  "
            f"{info.width}x{info.height} @ {info.fps:.3f}fps  {info.duration_s:.1f}s"
        )
        show_type = self._load_show_type()
        console.print(f"[bold]Show profile[/bold]: {show_type}")
        all_warnings: list[str] = []

        # Stage 1 — camera discovery (+ HITL)
        inventory = self._cached("01_cameras", lambda: discover_cameras(cfg, info, self.llm))
        inventory = self.hitl.review_cameras(inventory)
        all_warnings += inventory.get("warnings", [])

        # Stage 2 — transcription + motion + speaker mapping (+ HITL)
        wav = self.cache / "audio.wav"
        if not wav.exists() or self.force:
            extract_audio_wav(self.video, wav)
        transcript = self._cached("02_transcript", lambda: transcribe(cfg, wav))
        words = all_words(transcript)
        utterances = build_utterances(transcript)
        if not utterances:
            all_warnings.append("No speech detected — producing a wide-shot-only timeline")
        motion = self._cached_motion(info, inventory)
        mapping = self._cached(
            "03_speakers", lambda: map_speakers(cfg, utterances, inventory, motion, self.llm)
        )
        # inventory assignments may have been updated inside map_speakers on a fresh
        # run but not on a cached one — re-apply so both paths are identical
        sm = mapping["speaker_mapping"]
        if sm["host"]["tile"]:
            inventory["assignments"]["CAM_HOST_HERO"] = sm["host"]["tile"]
        if sm["guest"]["tile"]:
            inventory["assignments"]["CAM_GUEST_HERO"] = sm["guest"]["tile"]
        mapping = self.hitl.review_speakers(mapping, inventory)
        utterances = mapping["utterances"]
        all_warnings += mapping.get("warnings", [])

        # Stage 3 — narrative + visual events
        narrative = self._cached(
            "04_narrative",
            lambda: analyze_narrative(cfg, utterances, words, info.duration_s, self.llm),
        )
        visual = self._cached(
            "05_visual",
            lambda: detect_visual_events(cfg, self.video, inventory, utterances, motion, self.llm),
        )
        all_warnings += narrative.get("warnings", []) + visual.get("warnings", [])

        # Stage 4 — decision engine (deterministic, never cached: cheap + honors overrides)
        console.print("[bold cyan]stage 06_cuts: running decision engine[/bold cyan]")
        result = de.build_cut_list(
            cfg, info.duration_s, inventory, utterances, narrative, visual, show_type
        )
        result = de.apply_safety(cfg, result, words, narrative["events"], info.duration_s)
        result = de.enforce_wide_budget(cfg, result, inventory, show_type, info.duration_s)
        cuts = de.to_cut_dicts(result, inventory)
        all_warnings += result.get("warnings", [])

        # HITL — cut list approval gate
        cuts = self.hitl.review_cuts(cuts, cfg.output_dir, result.get("warnings", []))

        # Stage 5 — XML generation
        fcpxml_path = cfg.output_dir / cfg.output.get("fcpxml", "output.fcpxml")
        fcp7_path = cfg.output_dir / cfg.output.get("premiere_xml", "output_premiere.xml")
        write_fcpxml(fcpxml_path, info, inventory, cuts)
        write_fcp7_xml(fcp7_path, info, inventory, cuts)
        console.print(f"[green]wrote {fcpxml_path}[/green]")
        console.print(f"[green]wrote {fcp7_path}[/green]")

        # Stage 6 — validation
        validation = validate(
            cuts, words, info.duration_s,
            narrative["off_camera_segments"], [fcpxml_path, fcp7_path], info.fps,
        )
        all_warnings += validation["warnings"]
        status = "[green]PASSED[/green]" if validation["passed"] else "[red]FAILED[/red]"
        console.print(f"[bold]Validation[/bold]: {status}")
        for e in validation["errors"]:
            console.print(f"[red]✗ {e}[/red]")

        # Report
        report = build_report(
            info, show_type, inventory, mapping["speaker_mapping"], cuts,
            narrative["off_camera_segments"], validation, all_warnings,
            self.hitl.overrides,
            {"calls": self.llm.usage.calls, "vision_calls": self.llm.usage.vision_calls,
             "errors": self.llm.usage.errors[:20]},
        )
        report_path = cfg.output_dir / cfg.output.get("report", "editing_report.json")
        write_report(report, report_path)
        console.print(f"[green]wrote {report_path}[/green]")
        return report

    def _load_show_type(self) -> str:
        name = self.cfg.editing.get("show_type_file", "show_type.txt")
        for candidate in (Path(self.video).parent / name, self.cfg.root / name):
            if candidate.exists():
                return de.detect_show_type(candidate.read_text())
        console.print(f"[yellow]⚠ {name} not found — using generic show profile[/yellow]")
        return de.SHOW_GENERIC
