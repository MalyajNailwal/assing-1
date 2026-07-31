"""Human-in-the-loop checkpoints.

Three gates (each can be disabled in config or with --auto):
  cameras  — confirm/override detected camera roles & named assignments
  speakers — confirm/override host/guest mapping
  cuts     — review the full cut list (written to cuts_preview.json) before XML

Every override is recorded and lands in editing_report.json metadata.
"""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

console = Console()


class HITL:
    def __init__(self, enabled: bool, checkpoints: list[str], auto: bool = False):
        self.enabled = enabled and not auto
        self.checkpoints = set(checkpoints or [])
        self.overrides: list[dict] = []

    def _active(self, name: str) -> bool:
        return self.enabled and name in self.checkpoints

    # ------------------------------------------------------------- cameras
    def review_cameras(self, inventory: dict) -> dict:
        if not self._active("cameras"):
            return inventory
        table = Table(title="Stage 1 — Detected cameras (confirm before continuing)")
        for col in ("tile", "role", "person", "motion", "conf", "assignment"):
            table.add_column(col)
        assign_rev = {v: k for k, v in inventory["assignments"].items()}
        for t in inventory["tiles"]:
            table.add_row(
                t["id"], t["role"], (t.get("person_desc") or "-")[:40],
                f"{t['motion']:.2f}", f"{t.get('confidence', 0):.2f}",
                assign_rev.get(t["id"], "-"),
            )
        console.print(table)
        grid = inventory["grid"]
        console.print(f"Grid: {grid['rows']}x{grid['cols']} (confidence {grid['confidence']})")
        for w in inventory.get("warnings", []):
            console.print(f"[yellow]⚠ {w}[/yellow]")
        console.print(
            "\n[bold]Enter[/bold] to accept · or override, e.g. "
            "[cyan]CAM_WIDE=cam_5 CAM_HOST_HERO=cam_1 cam_2:EMPTY[/cyan] "
            "(NAME=tile reassigns, tile:ROLE changes a role) · [cyan]q[/cyan] aborts"
        )
        ans = Prompt.ask("cameras", default="").strip()
        if ans.lower() == "q":
            raise SystemExit("Aborted at camera checkpoint")
        for token in ans.split():
            if "=" in token:
                name, tile = token.split("=", 1)
                inventory["assignments"][name.strip().upper()] = tile.strip()
                self.overrides.append({"checkpoint": "cameras", "set": token})
            elif ":" in token:
                tile, role = token.split(":", 1)
                for t in inventory["tiles"]:
                    if t["id"] == tile.strip():
                        t["role"] = role.strip().upper()
                        self.overrides.append({"checkpoint": "cameras", "set": token})
        return inventory

    # ------------------------------------------------------------ speakers
    def review_speakers(self, mapping_result: dict, inventory: dict) -> dict:
        if not self._active("speakers"):
            return mapping_result
        sm = mapping_result["speaker_mapping"]
        console.print("\n[bold]Stage 2 — Speaker mapping[/bold]")
        console.print(f"  HOST  -> {sm['host']['tile']}   GUEST -> {sm['guest']['tile']}")
        console.print(f"  confidence={sm['confidence']:.2f}  reason: {sm['reason']}")
        for w in mapping_result.get("warnings", []):
            console.print(f"[yellow]⚠ {w}[/yellow]")
        sample = [u for u in mapping_result["utterances"] if u.get("speaker") in ("host", "guest")][:6]
        for u in sample:
            console.print(f"  [{u['speaker']}/{u['tile']}] {u['text'][:90]}")
        console.print("\n[bold]Enter[/bold] accepts · [cyan]swap[/cyan] swaps host/guest · "
                      "[cyan]host=cam_X guest=cam_Y[/cyan] sets tiles · [cyan]q[/cyan] aborts")
        ans = Prompt.ask("speakers", default="").strip().lower()
        if ans == "q":
            raise SystemExit("Aborted at speaker checkpoint")
        host, guest = sm["host"]["tile"], sm["guest"]["tile"]
        if ans == "swap":
            host, guest = guest, host
            self.overrides.append({"checkpoint": "speakers", "set": "swap"})
        else:
            for token in ans.split():
                if token.startswith("host="):
                    host = token.split("=", 1)[1]
                    self.overrides.append({"checkpoint": "speakers", "set": token})
                elif token.startswith("guest="):
                    guest = token.split("=", 1)[1]
                    self.overrides.append({"checkpoint": "speakers", "set": token})
        if (host, guest) != (sm["host"]["tile"], sm["guest"]["tile"]):
            sm["host"]["tile"], sm["guest"]["tile"] = host, guest
            inventory["assignments"]["CAM_HOST_HERO"] = host
            inventory["assignments"]["CAM_GUEST_HERO"] = guest
            for u in mapping_result["utterances"]:
                u["speaker"] = "host" if u["tile"] == host else "guest" if u["tile"] == guest else u["speaker"]
        return mapping_result

    # ---------------------------------------------------------------- cuts
    def review_cuts(self, cuts: list[dict], out_dir: Path, warnings: list[str]) -> list[dict]:
        if not self._active("cuts"):
            return cuts
        preview = out_dir / "cuts_preview.json"
        preview.write_text(json.dumps(cuts, indent=2))
        total = cuts[-1]["end"] if cuts else 0
        by_rule: dict[str, int] = {}
        for c in cuts:
            by_rule[c["rule"]] = by_rule.get(c["rule"], 0) + 1
        console.print(f"\n[bold]Stage 4 — Cut list[/bold]  ({len(cuts)} shots over {total:.0f}s)")
        console.print("  " + "  ".join(f"{k}:{v}" for k, v in sorted(by_rule.items())))
        for w in warnings:
            console.print(f"[yellow]⚠ {w}[/yellow]")
        table = Table(title="First 15 cuts (full list in cuts_preview.json)")
        for col in ("#", "start", "end", "shot", "rule", "reason"):
            table.add_column(col)
        for i, c in enumerate(cuts[:15], 1):
            table.add_row(
                str(i), f"{c['start']:.1f}", f"{c['end']:.1f}",
                "+".join(c["camera_labels"]), c["rule"], c["reason"][:45],
            )
        console.print(table)
        console.print(
            f"\nEdit [cyan]{preview}[/cyan] if needed (change cameras/kind/boundaries), then:\n"
            "[bold]Enter[/bold] accepts as-is · [cyan]reload[/cyan] re-reads the edited file · [cyan]q[/cyan] aborts"
        )
        while True:
            ans = Prompt.ask("cuts", default="").strip().lower()
            if ans == "q":
                raise SystemExit("Aborted at cut-list checkpoint")
            if ans == "reload":
                try:
                    cuts = json.loads(preview.read_text())
                    self.overrides.append({"checkpoint": "cuts", "set": "edited cuts_preview.json"})
                    console.print(f"[green]Reloaded {len(cuts)} cuts from file[/green]")
                except (json.JSONDecodeError, OSError) as e:
                    console.print(f"[red]Could not reload: {e}[/red]")
                    continue
            return cuts
