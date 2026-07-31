"""Shared geometry for both XML writers.

Each cut references the ONE SyncMaster file; the chosen camera tile is
isolated with crop + uniform scale + reposition. `viewport_params`
computes, for a tile rect and a target viewport (full canvas or half
canvas for SBS), the exact source crop and transform that makes the
tile fill the viewport with no neighbor-tile bleed and no distortion.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Placement:
    # crop of the source frame, percent from each edge (0-100)
    crop_l: float
    crop_r: float
    crop_t: float
    crop_b: float
    scale: float          # uniform scale factor (1.0 = 100%)
    tx: float             # translation of frame center, px in canvas space
    ty: float


def viewport_params(
    frame_w: int,
    frame_h: int,
    tile: tuple[int, int, int, int],
    viewport: tuple[float, float, float, float],
) -> Placement:
    """tile=(x,y,w,h) in source px; viewport=(x,y,w,h) in canvas px
    (canvas == source frame size for this pipeline)."""
    tx_, ty_, tw, th = tile
    vx, vy, vw, vh = viewport

    # 'fill' scale: cover the viewport, then shrink the source window to
    # exactly the viewport so nothing overlaps neighboring viewports.
    k = max(vw / tw, vh / th)
    src_w, src_h = vw / k, vh / k
    sx = tx_ + (tw - src_w) / 2
    sy = ty_ + (th - src_h) / 2

    crop_l = 100.0 * sx / frame_w
    crop_r = 100.0 * (frame_w - (sx + src_w)) / frame_w
    crop_t = 100.0 * sy / frame_h
    crop_b = 100.0 * (frame_h - (sy + src_h)) / frame_h

    # move the (scaled-about-frame-center) source-window center onto the
    # viewport center: T = Vc - Fc - k*(Sc - Fc)
    fcx, fcy = frame_w / 2, frame_h / 2
    scx, scy = sx + src_w / 2, sy + src_h / 2
    vcx, vcy = vx + vw / 2, vy + vh / 2
    tx = vcx - fcx - k * (scx - fcx)
    ty = vcy - fcy - k * (scy - fcy)

    return Placement(
        crop_l=round(crop_l, 4), crop_r=round(crop_r, 4),
        crop_t=round(crop_t, 4), crop_b=round(crop_b, 4),
        scale=round(k, 6), tx=round(tx, 2), ty=round(ty, 2),
    )


def placements_for_cut(cut: dict, inventory: dict) -> list[tuple[str, Placement]]:
    """[(tile_id, Placement)] — one entry for single shots, two for SBS."""
    w = inventory["video"]["width"]
    h = inventory["video"]["height"]
    rects = {t["id"]: tuple(t["rect"]) for t in inventory["tiles"]}
    if cut["kind"] == "sbs" and len(cut["cameras"]) >= 2:
        left, right = cut["cameras"][0], cut["cameras"][1]
        return [
            (left, viewport_params(w, h, rects[left], (0, 0, w / 2, h))),
            (right, viewport_params(w, h, rects[right], (w / 2, 0, w / 2, h))),
        ]
    cam = cut["cameras"][0]
    return [(cam, viewport_params(w, h, rects[cam], (0, 0, w, h)))]


def cut_comment(cut: dict, idx: int) -> str:
    labels = "+".join(cut["camera_labels"])
    extra = " | ".join(m["comment"] for m in cut.get("markers", []))
    base = (
        f" CUT {idx:03d} | {cut['start']:.2f}s -> {cut['end']:.2f}s | "
        f"{labels} ({'/'.join(cut['cameras'])}) | rule={cut['rule']} | {cut['reason']} "
    )
    return base + (f"| {extra} " if extra else "")
