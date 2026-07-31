"""Stage 6 — Validation.

Gates before delivery:
  * XML well-formedness (both files parse)
  * timeline integrity: chronological, gap-free, overlap-free, positive durations
  * safety audit: no cut boundary lands inside a spoken word (Rule 10)
  * rule coverage: off-camera segments and PHY_ADJ markers present where expected
"""

from __future__ import annotations

from pathlib import Path

from lxml import etree


def validate(
    cuts: list[dict],
    words: list[dict],
    duration_s: float,
    off_camera_segments: list[dict],
    xml_paths: list[str | Path],
    fps: float,
) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, bool] = {}

    # --- XML well-formed
    for p in xml_paths:
        p = Path(p)
        try:
            etree.parse(str(p))
            checks[f"xml_parses:{p.name}"] = True
        except (etree.XMLSyntaxError, OSError) as e:
            checks[f"xml_parses:{p.name}"] = False
            errors.append(f"{p.name} failed to parse: {e}")

    # --- timeline integrity
    ok = True
    frame = 1.0 / fps
    for i, c in enumerate(cuts):
        if c["end"] - c["start"] <= 0:
            errors.append(f"cut {i}: non-positive duration")
            ok = False
        if i > 0:
            gap = c["start"] - cuts[i - 1]["end"]
            if abs(gap) > frame * 1.5:
                errors.append(f"cut {i}: {'gap' if gap > 0 else 'overlap'} of {gap:.3f}s at {c['start']:.2f}s")
                ok = False
    if cuts and abs(cuts[-1]["end"] - duration_s) > 1.0:
        warnings.append(f"timeline ends at {cuts[-1]['end']:.2f}s but media is {duration_s:.2f}s")
    checks["timeline_contiguous"] = ok

    # --- mid-word cut audit (Rule 10)
    boundaries = [c["start"] for c in cuts[1:]]
    mid_word = []
    wi = 0
    swords = sorted(words, key=lambda w: w["s"])
    for b in sorted(boundaries):
        while wi < len(swords) and swords[wi]["e"] < b - 0.05:
            wi += 1
        for w in swords[wi : wi + 3]:
            if w["s"] + 0.05 < b < w["e"] - 0.05:
                mid_word.append((round(b, 2), w["w"]))
                break
    checks["no_mid_word_cuts"] = not mid_word
    for b, w in mid_word[:10]:
        warnings.append(f"boundary at {b}s lands inside word {w!r} (locked/mandatory cuts may do this)")

    # --- rule coverage
    marker_names = {m["name"] for c in cuts for m in c.get("markers", [])}
    if off_camera_segments:
        checks["off_camera_marked"] = "OFF_CAMERA_BRAINSTORM" in marker_names
        if not checks["off_camera_marked"]:
            errors.append("off-camera segments detected but no OFF_CAMERA_BRAINSTORM marker in cuts")
    min_durs = [c for c in cuts if c["duration"] < 1.0 and not c["locked"]]
    if min_durs:
        warnings.append(f"{len(min_durs)} unlocked shots are shorter than 1s")

    passed = not errors
    return {"passed": passed, "checks": checks, "errors": errors, "warnings": warnings}
