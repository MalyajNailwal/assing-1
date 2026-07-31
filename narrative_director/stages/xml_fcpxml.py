"""Stage 5a — FCPXML v1.10 writer (the brief's literal spec).

Times are expressed as rational seconds aligned to the frame duration,
so every cut is frame-exact. Editor guidance goes in XML comments,
<note> elements, and markers (PHY_ADJ_CUT / OFF_CAMERA_BRAINSTORM /
TECH_FAILURE are preserved as markers on their clips).
"""

from __future__ import annotations

from pathlib import Path

from lxml import etree

from ..media.probe import MediaInfo
from .xml_common import cut_comment, placements_for_cut


def _rt(frames: int, fps_num: int, fps_den: int) -> str:
    """Rational time string for a frame count."""
    if frames == 0:
        return "0s"
    return f"{frames * fps_den}/{fps_num}s"


def write_fcpxml(
    out_path: str | Path,
    info: MediaInfo,
    inventory: dict,
    cuts: list[dict],
    project_name: str = "AI Narrative Rough Cut",
) -> Path:
    fps_num, fps_den = info.fps_num, info.fps_den
    to_f = lambda t: round(t * info.fps)  # noqa: E731

    root = etree.Element("fcpxml", version="1.10")
    resources = etree.SubElement(root, "resources")
    etree.SubElement(
        resources, "format", id="r1",
        name=f"FFVideoFormat{info.height}p{round(info.fps * 100) / 100:g}".replace(".", ""),
        frameDuration=_rt(1, fps_num, fps_den),
        width=str(info.width), height=str(info.height),
    )
    total_frames = to_f(info.duration_s)
    asset = etree.SubElement(
        resources, "asset", id="r2", name=Path(info.path).stem,
        start="0s", duration=_rt(total_frames, fps_num, fps_den),
        hasVideo="1", hasAudio="1" if info.has_audio else "0", format="r1",
    )
    etree.SubElement(asset, "media-rep", kind="original-media",
                     src=Path(info.path).as_uri())

    library = etree.SubElement(root, "library")
    event = etree.SubElement(library, "event", name="AI Narrative Director")
    project = etree.SubElement(event, "project", name=project_name)
    seq_dur = to_f(cuts[-1]["end"]) if cuts else total_frames
    sequence = etree.SubElement(
        project, "sequence", format="r1",
        duration=_rt(seq_dur, fps_num, fps_den), tcStart="0s", tcFormat="NDF",
    )
    spine = etree.SubElement(sequence, "spine")

    for i, cut in enumerate(cuts, 1):
        f0, f1 = to_f(cut["start"]), to_f(cut["end"])
        if f1 <= f0:
            continue
        spine.append(etree.Comment(cut_comment(cut, i)))
        placements = placements_for_cut(cut, inventory)

        def clip_el(cam: str, pl, lane: int | None) -> etree._Element:
            el = etree.Element(
                "asset-clip", ref="r2",
                offset=_rt(f0, fps_num, fps_den),
                start=_rt(f0, fps_num, fps_den),  # synced multicam: source==timeline time
                duration=_rt(f1 - f0, fps_num, fps_den),
                name=f"{'+'.join(cut['camera_labels'])} [{cam}]",
                enabled="1",
            )
            if lane is not None:
                el.set("lane", str(lane))
            note = etree.SubElement(el, "note")
            note.text = f"{cut['rule']}: {cut['reason']}"
            crop = etree.SubElement(el, "adjust-crop", mode="trim")
            etree.SubElement(
                crop, "trim-rect",
                left=str(pl.crop_l), right=str(pl.crop_r),
                top=str(pl.crop_t), bottom=str(pl.crop_b),
            )
            etree.SubElement(
                el, "adjust-transform",
                position=f"{100.0 * pl.tx / info.width:.4f} {-100.0 * pl.ty / info.height:.4f}",
                scale=f"{pl.scale:.6f} {pl.scale:.6f}",
            )
            for m in cut.get("markers", []):
                mf = max(f0, min(f1 - 1, to_f(m["time"])))
                etree.SubElement(
                    el, "marker",
                    start=_rt(mf, fps_num, fps_den),
                    duration=_rt(1, fps_num, fps_den),
                    value=m["comment"],
                )
            return el

        primary = clip_el(placements[0][0], placements[0][1], None)
        for cam, pl in placements[1:]:
            primary.append(clip_el(cam, pl, 1))
        spine.append(primary)

    out_path = Path(out_path)
    out_path.write_bytes(
        etree.tostring(
            root, xml_declaration=True, encoding="UTF-8",
            pretty_print=True, doctype="<!DOCTYPE fcpxml>",
        )
    )
    return out_path
