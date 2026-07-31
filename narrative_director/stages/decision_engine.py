"""Stage 4 — Editorial Decision Engine.

Deliberately LLM-free: upstream stages produce *facts* (who speaks when,
narrative events, physical adjustments, freezes); this stage applies the
editorial rulebook as ordered, deterministic timeline transformations.
Same inputs => same cut list, every run.

Rule application order (later passes may override earlier ones):
  1.  base speaker timeline (Rule 1)
  2.  show-specific overlays (intro/outro/opening-question SBS, shared laughter)
  3.  rapid-dialogue wide/SBS (Rule 4)
  4.  monologue alternate angles (Rule 5)
  5.  listener reactions (Rule 2) — blocked inside emotional holds (Rules 1/6)
  6.  refresh wide shots (Rule 3)
  7.  physical-adjustment cutaways (Rule 7, locked)
  8.  frozen-camera avoidance (Rule 8, locked)
  9.  off-camera segments (Rule 9, strongest lock)
  10. safety pass: snap to word gaps, avoid laughter spans, min-shot merge (Rule 10)
  11. wide-usage budget for Nav Thethi (show rule)
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from ..config import Config

SHOW_NAV = "nav_thethi"
SHOW_MATURITY = "maturity_code"
SHOW_GENERIC = "generic"

EMO_TYPES = {"emotional", "story", "important_statement"}
EMO_MIN_INTENSITY = 0.55


@dataclass
class Shot:
    start: float
    end: float
    kind: str  # "single" | "sbs"
    cameras: list[str]  # tile ids ("cam_1"); sbs => [left, right]
    rule: str
    reason: str
    locked: bool = False
    markers: list[dict] = field(default_factory=list)

    @property
    def dur(self) -> float:
        return self.end - self.start


def detect_show_type(text: str) -> str:
    t = text.lower()
    if "nav" in t and "thethi" in t:
        return SHOW_NAV
    if "maturity" in t:
        return SHOW_MATURITY
    return SHOW_GENERIC


# ------------------------------------------------------------- timeline ops
def _overlay(timeline: list[Shot], new: Shot, force: bool = False) -> bool:
    """Insert `new`, clipping whatever it overlaps. Returns False (no-op) if it
    would overlap a locked shot and force is False."""
    if new.dur <= 0.05:
        return False
    if not force and any(s.locked and s.start < new.end and s.end > new.start for s in timeline):
        return False

    def markers_in(s: Shot, a: float, b: float) -> list[dict]:
        return [m for m in s.markers if a <= m["time"] < b]

    out: list[Shot] = []
    for s in timeline:
        if s.end <= new.start or s.start >= new.end:
            out.append(s)
            continue
        # markers inside the overridden span migrate to the new shot —
        # mandatory markers (PHY_ADJ_CUT etc.) must never silently vanish
        new.markers.extend(m for m in markers_in(s, new.start, new.end) if m not in new.markers)
        if s.start < new.start:
            left = copy.deepcopy(s)
            left.end = new.start
            left.markers = markers_in(s, s.start, new.start)
            out.append(left)
        if s.end > new.end:
            right = copy.deepcopy(s)
            right.start = new.end
            right.markers = markers_in(s, new.end, s.end)
            out.append(right)
    out.append(new)
    out.sort(key=lambda s: s.start)
    timeline[:] = out
    return True


def _merge_adjacent(timeline: list[Shot]) -> None:
    out: list[Shot] = []
    for s in timeline:
        p = out[-1] if out else None
        if (
            p
            and not p.locked
            and not s.locked
            and p.kind == s.kind
            and p.cameras == s.cameras
            and abs(p.end - s.start) < 0.05
        ):
            p.end = s.end
            p.markers.extend(s.markers)
        else:
            out.append(s)
    timeline[:] = out


def _shot_at(timeline: list[Shot], t: float) -> Shot | None:
    for s in timeline:
        if s.start <= t < s.end:
            return s
    return None


# ---------------------------------------------------------------- main build
def build_cut_list(
    cfg: Config,
    duration_s: float,
    inventory: dict,
    utterances: list[dict],
    narrative: dict,
    visual: dict,
    show_type: str,
) -> dict:
    ed = cfg.editing
    assign = inventory["assignments"]
    host = assign.get("CAM_HOST_HERO")
    guest = assign.get("CAM_GUEST_HERO")
    wide = assign.get("CAM_WIDE")
    heroes = [c for c in (host, guest) if c]
    fallback = wide or host or guest or (inventory["tiles"][0]["id"] if inventory["tiles"] else "cam_1")
    warnings: list[str] = list(narrative.get("warnings", [])) + list(visual.get("warnings", []))

    def hero_of(speaker_tile: str | None) -> str:
        return speaker_tile if speaker_tile in heroes else fallback

    def listener_of(tile: str) -> str | None:
        if tile == host:
            return guest
        if tile == guest:
            return host
        return None

    def neutral_shot(start: float, end: float, rule: str, reason: str, locked=False) -> Shot:
        """Wide when available; else SBS of both heroes (the only 'wider
        composition' possible without a wide cam); else the fallback hero."""
        if wide:
            return Shot(start, end, "single", [wide], rule, reason, locked)
        if host and guest:
            return Shot(start, end, "sbs", [host, guest], rule, reason, locked)
        return Shot(start, end, "single", [fallback], rule, reason, locked)

    events = narrative["events"]
    emo_holds = [
        (e["start"], e["end"])
        for e in events
        if e["type"] in EMO_TYPES and e["intensity"] >= EMO_MIN_INTENSITY
    ]

    def in_hold(a: float, b: float) -> bool:
        return any(s < b and e > a for s, e in emo_holds)

    # ---- pass 1: base speaker timeline (Rule 1) ------------------------------
    timeline: list[Shot] = []
    cursor = 0.0
    prev_cam = fallback
    turns: list[dict] = []
    for u in utterances:
        cam = hero_of(u.get("tile"))
        if turns and turns[-1]["cam"] == cam and u["start"] - turns[-1]["end"] < 2.0:
            turns[-1]["end"] = u["end"]
        else:
            turns.append({"start": u["start"], "end": u["end"], "cam": cam, "speaker": u.get("speaker", "?")})
    # Nav Thethi is hero-first: hold the previous hero through pauses much
    # longer before falling back to a neutral/wide shot.
    pause_neutral_after = 10.0 if show_type == SHOW_NAV else 4.0
    for t in turns:
        if t["start"] > cursor + 0.01:
            # silence gap: short => stay on previous camera; long => neutral
            if t["start"] - cursor > pause_neutral_after:
                timeline.append(neutral_shot(cursor, t["start"], "SPEAKER_RULE", "long pause — neutral shot"))
            else:
                timeline.append(Shot(cursor, t["start"], "single", [prev_cam], "SPEAKER_RULE", "hold through pause"))
        timeline.append(Shot(t["start"], t["end"], "single", [t["cam"]], "SPEAKER_RULE", f"{t['speaker']} speaking"))
        prev_cam = t["cam"]
        cursor = t["end"]
    if cursor < duration_s:
        timeline.append(neutral_shot(cursor, duration_s, "SPEAKER_RULE", "tail"))
    _merge_adjacent(timeline)

    # ---- pass 2: show-specific overlays --------------------------------------
    if show_type == SHOW_MATURITY and host and guest:
        first_q = next((e for e in events if e["type"] == "question"), None)
        sbs_end = (first_q["end"] if first_q else min(20.0, duration_s)) + 2.0
        _overlay(timeline, Shot(0.0, min(sbs_end, duration_s), "sbs", [host, guest],
                                "SHOW_RULE", "Maturity Code: intro + opening question in SBS"))
        if duration_s > 30:
            _overlay(timeline, Shot(max(0.0, duration_s - 12.0), duration_s, "sbs", [host, guest],
                                    "SHOW_RULE", "Maturity Code: outro in SBS"))
        for e in events:  # shared laughter => SBS
            if e["type"] == "laughter" and e["intensity"] >= 0.5:
                s = max(0.0, e["start"] - 0.3)
                _overlay(timeline, Shot(s, min(s + 4.0, duration_s), "sbs", [host, guest],
                                        "SHOW_RULE", "shared laughter — SBS"))

    # ---- pass 3: rapid dialogue (Rule 4) --------------------------------------
    rapid_turn = float(ed.get("rapid_turn_s", 4.0))
    rapid_n = int(ed.get("rapid_turns_window", 4))
    i = 0
    while i + rapid_n <= len(turns):
        window = turns[i : i + rapid_n]
        alternating = all(window[k]["cam"] != window[k + 1]["cam"] for k in range(len(window) - 1))
        if alternating and all(t["end"] - t["start"] < rapid_turn for t in window):
            j = i + rapid_n
            while j < len(turns) and turns[j]["end"] - turns[j]["start"] < rapid_turn and turns[j]["cam"] != turns[j - 1]["cam"]:
                j += 1
            s, e = window[0]["start"], turns[j - 1]["end"]
            if not in_hold(s, e):
                _overlay(timeline, neutral_shot(s, e, "DIALOGUE_RULE", "rapid back-and-forth — wider composition"))
            i = j
        else:
            i += 1

    # ---- pass 4: long monologues (Rule 5) --------------------------------------
    mono_min = float(ed.get("monologue_min_s", 30.0))
    alt_every = float(ed.get("monologue_alt_every_s", 25.0))
    alternates: dict[str, list[str]] = inventory.get("alternates", {})
    for t in turns:
        if t["end"] - t["start"] < mono_min:
            continue
        # variety pool: speaker's own ALT angles (stay with the storyteller),
        # then the listener, then the wide
        pool = list(alternates.get(t["cam"], []))
        if listener_of(t["cam"]):
            pool.append(listener_of(t["cam"]))
        if wide:
            pool.append(wide)
        if not pool:
            continue
        at = t["start"] + alt_every
        k = 0
        while at + 4.0 < t["end"] - 5.0:
            alt_cam = pool[k % len(pool)]
            # during emotional holds, an ALT angle of the SAME speaker is
            # allowed (stays on the storyteller); cutting away is not
            same_person = alt_cam in alternates.get(t["cam"], [])
            if same_person or not in_hold(at, at + 4.0):
                _overlay(timeline, Shot(at, at + 4.0, "single", [alt_cam],
                                        "MONOLOGUE_RULE", "long monologue — alternate angle"))
                k += 1
            at += alt_every

    # ---- pass 5: listener reactions (Rule 2) ------------------------------------
    r_min = float(ed.get("reaction_min_s", 3.0))
    r_max = float(ed.get("reaction_max_s", 5.0))
    last_reaction_end = -999.0
    for ev in visual.get("reactions", []):
        t0 = ev["time"]
        if t0 - last_reaction_end < 15.0:  # rate limit reaction cutaways
            continue
        cur = _shot_at(timeline, t0)
        if cur is None or cur.kind != "single" or ev["camera"] in cur.cameras:
            continue  # only cut TO the listener while someone else holds the frame
        if in_hold(t0, t0 + r_min):
            continue  # emotional priority: stay on the storyteller
        dur = r_max if ev["detail"] in ("laugh", "moved") else r_min
        end = min(t0 + dur, duration_s)
        if _overlay(timeline, Shot(t0, end, "single", [ev["camera"]],
                                   "REACTION_RULE", f"listener {ev['detail']}")):
            last_reaction_end = end

    # ---- pass 6: refresh rule (Rule 3) -------------------------------------------
    refresh_after = float(ed.get("refresh_after_s", 45.0))
    refresh_len = float(ed.get("refresh_wide_s", 3.0))
    if wide:
        timeline.sort(key=lambda s: s.start)
        interesting = sorted(
            [e["start"] for e in events if e["type"] != "silence"]
            + [s.start for s in timeline]
        )
        t = refresh_after
        while t < duration_s - refresh_len - 2.0:
            recent = [x for x in interesting if t - refresh_after < x <= t]
            if not recent and not in_hold(t, t + refresh_len):
                cur = _shot_at(timeline, t)
                if cur and wide not in cur.cameras:
                    _overlay(timeline, Shot(t, t + refresh_len, "single", [wide],
                                            "REFRESH_RULE", "45s without events — establishing wide"))
                    interesting.append(t)
                    interesting.sort()
            t += 5.0

    # ---- pass 7: physical adjustments (Rule 7, mandatory, locked) ------------------
    for ev in visual.get("phys_adjustments", []):
        t0 = ev["time"]
        cur = _shot_at(timeline, t0)
        if cur is None or ev["camera"] not in cur.cameras or cur.kind == "sbs":
            continue  # offender not on screen — nothing to cut away from
        away = listener_of(ev["camera"]) or wide
        if not away:  # last resort: a different angle of the same person
            away = next(iter(inventory.get("alternates", {}).get(ev["camera"], [])), None)
        if not away:
            warnings.append(f"PHY_ADJ at {t0:.1f}s ({ev['detail']}) but no alternate camera available")
            continue
        shot = Shot(
            max(0.0, t0 - 0.2), min(t0 + 3.0, duration_s), "single", [away],
            "PHY_ADJ_CUT", f"speaker {ev['detail']} — mandatory cutaway", locked=True,
        )
        shot.markers.append({"time": t0, "name": "PHY_ADJ_CUT", "comment": f"PHY_ADJ_CUT: {ev['detail']} on {ev['camera']}"})
        _overlay(timeline, shot, force=True)

    # ---- pass 8: frozen cameras (Rule 8, locked) -------------------------------------
    for fz in visual.get("freezes", []):
        for s in [s for s in timeline if s.start < fz["end"] and s.end > fz["start"]]:
            if fz["camera"] not in s.cameras:
                continue
            alt = next((c for c in [listener_of(fz["camera"]), wide, host, guest] if c and c != fz["camera"]), None)
            if not alt:
                warnings.append(f"Camera {fz['camera']} frozen {fz['start']:.1f}-{fz['end']:.1f}s; no alternate")
                continue
            shot = Shot(max(s.start, fz["start"]), min(s.end, fz["end"]), "single", [alt],
                        "TECH_FAILURE", f"{fz['camera']} frozen — switched to {alt}", locked=True)
            shot.markers.append({"time": shot.start, "name": "TECH_FAILURE",
                                 "comment": f"TECH_FAILURE: {fz['camera']} frozen"})
            _overlay(timeline, shot, force=True)

    # ---- pass 9: off-camera segments (Rule 9, strongest) --------------------------------
    for seg in narrative.get("off_camera_segments", []):
        shot = neutral_shot(seg["start"], seg["end"], "OFF_CAMERA_BRAINSTORM",
                            "'stop rolling' detected — editing suspended", locked=True)
        shot.markers.append({"time": seg["start"], "name": "OFF_CAMERA_BRAINSTORM",
                             "comment": "OFF_CAMERA_BRAINSTORM: remove before publish"})
        _overlay(timeline, shot, force=True)

    timeline.sort(key=lambda s: s.start)
    _merge_adjacent(timeline)
    return {"timeline": timeline, "turns": turns, "warnings": warnings}


# ------------------------------------------------------------- safety passes
def apply_safety(
    cfg: Config,
    result: dict,
    words: list[dict],
    events: list[dict],
    duration_s: float,
) -> dict:
    """Rule 10: snap cut points to word gaps, keep out of laughter spans,
    enforce minimum shot length."""
    ed = cfg.editing
    timeline: list[Shot] = result["timeline"]
    min_shot = float(ed.get("min_shot_s", 2.0))
    gap_needed = float(ed.get("word_gap_safe_s", 0.25))
    laughs = [(e["start"], e["end"]) for e in events if e["type"] == "laughter"]

    # candidate safe times = centers of word gaps >= gap_needed, outside laughter
    safe: list[float] = [0.0, duration_s]
    for a, b in zip(words, words[1:]):
        if b["s"] - a["e"] >= gap_needed:
            mid = (a["e"] + b["s"]) / 2
            if not any(ls - 0.2 < mid < le + 0.2 for ls, le in laughs):
                safe.append(mid)
    safe.sort()

    def snap(t: float, window: float) -> float:
        best, bd = t, window + 1
        for s in safe:
            d = abs(s - t)
            if d < bd:
                best, bd = s, d
            if s > t + window:
                break
        return best if bd <= window else t

    moved = 0
    for prev, cur in zip(timeline, timeline[1:]):
        window = 0.3 if (prev.locked or cur.locked) else 0.7
        t = snap(cur.start, window)
        lo = prev.start + 0.5
        hi = cur.end - 0.5
        if lo < t < hi and abs(t - cur.start) > 0.01:
            prev.end = cur.start = t
            moved += 1

    # enforce minimum shot duration by absorbing runts into a neighbor
    changed = True
    while changed:
        changed = False
        for i, s in enumerate(timeline):
            if s.dur >= min_shot or s.locked or len(timeline) == 1:
                continue
            prev = timeline[i - 1] if i > 0 else None
            nxt = timeline[i + 1] if i + 1 < len(timeline) else None
            absorber = prev if (prev and not prev.locked) else nxt if (nxt and not nxt.locked) else None
            if absorber is None:
                continue
            if absorber is prev:
                prev.end = s.end
            else:
                nxt.start = s.start
            absorber.markers.extend(s.markers)
            timeline.pop(i)
            changed = True
            break
    _merge_adjacent(timeline)
    result["safety"] = {"snapped_boundaries": moved, "safe_points": len(safe)}
    return result


def enforce_wide_budget(cfg: Config, result: dict, inventory: dict, show_type: str, duration_s: float) -> dict:
    """Nav Thethi show rule: wide usage below `editing.wide_budget` of runtime."""
    if show_type != SHOW_NAV:
        return result
    wide = inventory["assignments"].get("CAM_WIDE")
    if not wide:
        return result
    budget = float(cfg.editing.get("wide_budget", 0.20)) * duration_s
    timeline: list[Shot] = result["timeline"]
    host = inventory["assignments"].get("CAM_HOST_HERO")

    def wide_time() -> float:
        return sum(s.dur for s in timeline if not s.locked and s.kind == "single" and s.cameras == [wide])

    # sacrifice longest non-mandatory wide shots first (refresh/dialogue holds)
    while wide_time() > budget:
        candidates = [s for s in timeline
                      if not s.locked and s.cameras == [wide] and s.rule in ("DIALOGUE_RULE", "SPEAKER_RULE", "REFRESH_RULE")]
        if not candidates:
            result["warnings"].append("Wide budget exceeded but remaining wide shots are mandatory")
            break
        victim = max(candidates, key=lambda s: s.dur)
        victim.cameras = [host or wide]
        victim.rule = "SPEAKER_RULE"
        victim.reason = "converted from wide (Nav Thethi wide budget)"
        _merge_adjacent(timeline)
        if victim.cameras == [wide]:
            break
    return result


def to_cut_dicts(result: dict, inventory: dict) -> list[dict]:
    label_of = {v: k for k, v in inventory["assignments"].items()}
    cuts = []
    for s in result["timeline"]:
        cuts.append(
            {
                "start": round(s.start, 3),
                "end": round(s.end, 3),
                "duration": round(s.dur, 3),
                "kind": s.kind,
                "cameras": s.cameras,
                "camera_labels": [label_of.get(c, c) for c in s.cameras],
                "rule": s.rule,
                "reason": s.reason,
                "locked": s.locked,
                "markers": s.markers,
            }
        )
    return cuts
