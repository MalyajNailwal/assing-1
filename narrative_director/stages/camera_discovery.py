"""Stage 1 — Camera discovery.

Finds the SyncMaster tile grid geometrically (no hardcoded coordinates),
measures per-tile activity, then asks the vision LLM to classify camera
roles (HERO / WIDE / EMPTY) and describe the person on each camera.
Host-vs-guest is provisional here; Stage 2 finalizes it from speech and
the human confirms it at a HITL checkpoint.
"""

from __future__ import annotations

import numpy as np
import cv2

from ..config import Config
from ..llm import LLMClient
from ..media.extract import contact_sheet, crop_tile, png_bytes, resize_width, sample_frames
from ..media.probe import MediaInfo

MAX_GRID = 4  # search grids up to 4x4

ROLE_PROMPT = """You are analyzing a multicam podcast "SyncMaster" recording. The image is a contact sheet: each cell is one camera tile, labeled cam_1, cam_2, ... in reading order.

For EACH camera tile, classify:
- "role": one of "HERO" (close-up/medium of ONE person), "WIDE" (shows two or more people / the whole set), "EMPTY" (black, color bars, no signal, or an empty chair/no person), "OTHER" (screen share, graphics, b-roll)
- "person_id": a letter ("A", "B", ...) — tiles showing the SAME person must share the same letter; null if no person. Multiple camera angles of one person are common.
- "tightness": "close" | "medium" | "wide" — how tight the framing on the person is
- "person_count": integer number of visible people
- "person_desc": short visual description of the person(s) (clothing, hair, position) or "" if none
- "likely_host": true/false/null — your best guess whether this person is the show HOST (hosts often face the guest, have branded mics/notes). Use null when unsure.
- "confidence": 0.0-1.0

Reply with ONLY a JSON object:
{"tiles": [{"id": "cam_1", "role": "...", "person_id": null, "tightness": "medium", "person_count": 0, "person_desc": "...", "likely_host": null, "confidence": 0.0}, ...],
 "notes": "anything unusual about the layout"}"""

GRID_FALLBACK_PROMPT = """This frame is a multicam "SyncMaster" recording: several camera feeds composited into one frame, usually in a uniform grid. Tell me the layout.

Reply ONLY with JSON: {"rows": <int>, "cols": <int>, "notes": "..."}"""


# --------------------------------------------------------------- grid finding
def _boundary_scores(edge_mean: np.ndarray, rows: int, cols: int) -> list[float]:
    """Edge energy along EACH internal boundary of a rows x cols uniform grid,
    normalized against the global mean edge energy. A real grid line shows
    strong persistent edges along its entire length."""
    h, w = edge_mean.shape
    base = float(edge_mean.mean()) + 1e-6
    scores = []
    for r in range(1, rows):
        y = round(h * r / rows)
        band = edge_mean[max(0, y - 2) : y + 3, :]
        scores.append(float(band.max(axis=0).mean()) / base)
    for c in range(1, cols):
        x = round(w * c / cols)
        band = edge_mean[:, max(0, x - 2) : x + 3]
        scores.append(float(band.max(axis=1).mean()) / base)
    return scores


def detect_grid(frames: list[np.ndarray], llm: LLMClient | None = None) -> tuple[int, int, float]:
    """Return (rows, cols, confidence). Tries geometric evidence first,
    falls back to the vision LLM when ambiguous."""
    valid = [f for f in frames if f is not None]
    if not valid:
        raise RuntimeError("No decodable frames for grid detection")
    edges = []
    for f in valid:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edges.append(np.abs(gx) + np.abs(gy))
    edge_mean = np.mean(edges, axis=0)

    h, w = edge_mean.shape
    candidates: list[tuple[int, int, float]] = []  # (rows, cols, weakest-boundary score)
    for rows in range(1, MAX_GRID + 1):
        for cols in range(1, MAX_GRID + 1):
            if rows == 1 and cols == 1:
                continue
            if w / cols < 120 or h / rows < 90:  # tiles implausibly small
                continue
            scores = _boundary_scores(edge_mean, rows, cols)
            if scores:
                candidates.append((rows, cols, min(scores)))

    # A candidate is credible only if EVERY claimed boundary shows strong
    # evidence (weakest boundary >= 2x baseline). Among credible candidates,
    # take the finest grid — min-scoring rejects over-segmentation naturally,
    # since a fictional boundary line scores near baseline.
    strong = [c for c in candidates if c[2] >= 2.0]
    if strong:
        rows, cols, smin = max(strong, key=lambda c: (c[0] * c[1], c[2]))
        return rows, cols, min(1.0, 0.5 + smin / 6.0)

    # Ambiguous — ask the vision LLM if available, else single tile.
    if llm is not None:
        img = resize_width(valid[len(valid) // 2], 1024)
        ans = llm.ask_json(GRID_FALLBACK_PROMPT, images_png=[png_bytes(img)])
        rows, cols = int(ans.get("rows", 1)), int(ans.get("cols", 1))
        if 1 <= rows <= MAX_GRID and 1 <= cols <= MAX_GRID:
            return rows, cols, 0.6
    weak = max(candidates, key=lambda c: c[2], default=(1, 1, 0.0))
    if weak[2] > 1.3:
        return weak[0], weak[1], 0.3
    return 1, 1, 0.3


def detect_content_tiles(frames: list[np.ndarray]) -> list[tuple[int, int, int, int]] | None:
    """Freeform layouts: tiles are content regions on a uniform background
    (white/black margins). Returns rects in reading order, or None if the
    layout looks like a margin-less uniform grid (caller falls back)."""
    stack = np.stack([f.astype(np.float32) for f in frames])
    mean = stack.mean(axis=0)
    motion = np.abs(np.diff(stack, axis=0)).mean(axis=(0, 3))

    h, w = mean.shape[:2]
    border = np.concatenate(
        [mean[0:4].reshape(-1, 3), mean[-4:].reshape(-1, 3),
         mean[:, 0:4].reshape(-1, 3), mean[:, -4:].reshape(-1, 3)]
    )
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(mean - bg, axis=2)
    mask = ((dist > 30) | (motion > 2.0)).astype(np.uint8) * 255

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    boxes = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if bw < w * 0.08 or bh < h * 0.08 or area < w * h * 0.01:
            continue
        boxes.append((int(x), int(y), int(bw), int(bh)))
    if len(boxes) < 2:
        return None
    union = sum(bw * bh for _, _, bw, bh in boxes)
    biggest = max(bw * bh for _, _, bw, bh in boxes)
    if biggest > 0.85 * w * h:  # one giant region == margin-less grid
        return None
    if union > 0.98 * w * h:
        return None
    # reading order: band rows by y-center, then left-to-right
    boxes.sort(key=lambda b: (round((b[1] + b[3] / 2) / (h * 0.22)), b[0]))
    return boxes


def tile_rects(width: int, height: int, rows: int, cols: int) -> list[tuple[int, int, int, int]]:
    rects = []
    for r in range(rows):
        for c in range(cols):
            x0, x1 = round(width * c / cols), round(width * (c + 1) / cols)
            y0, y1 = round(height * r / rows), round(height * (r + 1) / rows)
            rects.append((x0, y0, x1 - x0, y1 - y0))
    return rects


# ------------------------------------------------------------- tile analysis
def _tile_stats(frames: list[np.ndarray], rect) -> dict:
    """Motion energy + luma stats for one tile across sampled frames."""
    crops = [cv2.cvtColor(crop_tile(f, rect), cv2.COLOR_BGR2GRAY) for f in frames if f is not None]
    if len(crops) < 2:
        return {"motion": 0.0, "luma_mean": 0.0, "luma_std": 0.0}
    diffs = [float(np.mean(cv2.absdiff(a, b))) for a, b in zip(crops, crops[1:])]
    return {
        "motion": float(np.mean(diffs)),
        "luma_mean": float(np.mean([c.mean() for c in crops])),
        "luma_std": float(np.mean([c.std() for c in crops])),
    }


_FACE_CASCADE = None


def _face_cascade():
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        _FACE_CASCADE = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    return _FACE_CASCADE


def detect_face(tile_bgr: np.ndarray) -> list[int] | None:
    """Largest face rect [x,y,w,h] in tile coords, or None."""
    gray = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade().detectMultiScale(gray, 1.15, 5, minSize=(40, 40))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return [int(x), int(y), int(w), int(h)]


# ---------------------------------------------------------------- main stage
def discover_cameras(cfg: Config, info: MediaInfo, llm: LLMClient) -> dict:
    n = int(cfg.analysis.get("grid_sample_frames", 16))
    times = [info.duration_s * (i + 0.5) / n for i in range(n)]
    frames = sample_frames(info.path, times)
    valid = [f for f in frames if f is not None]
    if not valid:
        raise RuntimeError("Could not decode any frames from the video")

    # freeform layout first (tiles on a background), uniform grid as fallback
    rects = detect_content_tiles(valid)
    if rects is not None:
        rows, cols, grid_conf = 0, 0, 0.9
        layout = "freeform"
    else:
        rows, cols, grid_conf = detect_grid(valid, llm)
        rects = tile_rects(info.width, info.height, rows, cols)
        layout = "grid"

    tiles = []
    mid = valid[len(valid) // 2]
    for i, rect in enumerate(rects):
        stats = _tile_stats(valid, rect)
        # face box: try several frames (hands/turned heads defeat single-frame detection)
        face = None
        for f in valid[len(valid) // 2 :: 2] + valid[: len(valid) // 2 : 2]:
            face = detect_face(crop_tile(f, rect))
            if face:
                break
        # empty heuristics: nearly black, or nearly static AND featureless
        empty = (stats["luma_mean"] < 12 and stats["luma_std"] < 8) or (
            stats["motion"] < 0.15 and stats["luma_std"] < 6
        )
        tiles.append(
            {
                "id": f"cam_{i + 1}",
                "rect": list(rect),
                "motion": round(stats["motion"], 3),
                "luma_mean": round(stats["luma_mean"], 1),
                "face_rect": face,
                "empty_heuristic": bool(empty),
            }
        )

    # Vision LLM classifies roles from an annotated contact sheet
    sheet = contact_sheet(
        [crop_tile(mid, tuple(t["rect"])) for t in tiles],
        [t["id"] for t in tiles],
        cols=cols or 3,
    )
    ans = llm.ask_json(ROLE_PROMPT, images_png=[png_bytes(sheet)])
    llm_tiles = {t.get("id"): t for t in ans.get("tiles", [])}

    warnings = []
    for t in tiles:
        lt = llm_tiles.get(t["id"], {})
        role = str(lt.get("role", "OTHER")).upper()
        if t["empty_heuristic"] and role not in ("EMPTY",):
            # trust pixels over the LLM for dead feeds, but log it
            if t["luma_mean"] < 12:
                warnings.append(f"{t['id']}: LLM said {role} but tile is black — marking EMPTY")
                role = "EMPTY"
        t["role"] = role
        t["person_id"] = lt.get("person_id")
        t["tightness"] = lt.get("tightness", "medium")
        t["person_count"] = int(lt.get("person_count", 0) or 0)
        t["person_desc"] = lt.get("person_desc", "")
        t["likely_host"] = lt.get("likely_host", None)
        t["confidence"] = float(lt.get("confidence", 0.5) or 0.5)

    heroes = [t for t in tiles if t["role"] == "HERO"]
    wides = [t for t in tiles if t["role"] == "WIDE"]
    if not heroes:
        warnings.append("No HERO cameras detected — check camera mapping at the HITL checkpoint")
    if not wides:
        warnings.append("No WIDE camera detected — refresh/dialogue rules will fall back to heroes/SBS")

    # Multiple angles of the same person: the tightest, highest-confidence
    # angle is that person's primary HERO; the rest become ALT angles.
    tightness_rank = {"close": 0, "medium": 1, "wide": 2}
    by_person: dict[str, list[dict]] = {}
    for t in heroes:
        by_person.setdefault(t.get("person_id") or t["id"], []).append(t)
    alternates: dict[str, list[str]] = {}
    primaries: list[dict] = []
    for pid, group in by_person.items():
        group.sort(key=lambda t: (tightness_rank.get(t.get("tightness"), 1), -t["confidence"]))
        primary, alts = group[0], group[1:]
        for a in alts:
            a["role"] = "ALT"
        alternates[primary["id"]] = [a["id"] for a in alts]
        primaries.append(primary)
        if alts:
            warnings.append(
                f"person {pid}: {len(group)} angles — {primary['id']} is primary HERO, "
                f"{[a['id'] for a in alts]} kept as ALT angles"
            )

    # Provisional named assignment (finalized in Stage 2 + HITL)
    assignments: dict[str, str] = {}
    host_first = sorted(primaries, key=lambda t: (t.get("likely_host") is not True, -t["confidence"]))
    if host_first:
        assignments["CAM_HOST_HERO"] = host_first[0]["id"]
    if len(host_first) > 1:
        assignments["CAM_GUEST_HERO"] = host_first[1]["id"]
    if wides:
        assignments["CAM_WIDE"] = max(wides, key=lambda t: t["confidence"])["id"]

    return {
        "grid": {"rows": rows, "cols": cols, "layout": layout, "confidence": round(float(grid_conf), 2)},
        "alternates": alternates,
        "video": {"width": info.width, "height": info.height, "fps": info.fps},
        "tiles": tiles,
        "assignments": assignments,
        "llm_notes": ans.get("notes", ""),
        "warnings": warnings,
    }
