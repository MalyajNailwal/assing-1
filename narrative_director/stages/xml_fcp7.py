"""Stage 5b — FCP7 xmeml writer (Premiere Pro's native XML interchange).

Two video tracks (V2 used only for SBS right-half clips), one audio track
carrying the SyncMaster program audio uncut. Each clip gets Crop + Basic
Motion filters that isolate its camera tile, clipitem comments carrying
the editorial rationale, and sequence markers for PHY_ADJ_CUT /
OFF_CAMERA_BRAINSTORM / TECH_FAILURE.
"""

from __future__ import annotations

from pathlib import Path

from lxml import etree

from ..media.probe import MediaInfo
from .xml_common import cut_comment, placements_for_cut


def _el(parent, tag, text=None, **attrs):
    e = etree.SubElement(parent, tag, **{k: str(v) for k, v in attrs.items()})
    if text is not None:
        e.text = str(text)
    return e


def _rate(parent, timebase: int, ntsc: bool):
    r = _el(parent, "rate")
    _el(r, "timebase", timebase)
    _el(r, "ntsc", "TRUE" if ntsc else "FALSE")
    return r


def _param(effect, pid, name, value, vmin=None, vmax=None):
    p = _el(effect, "parameter", authoringApp="PremierePro")
    _el(p, "parameterid", pid)
    _el(p, "name", name)
    if vmin is not None:
        _el(p, "valuemin", vmin)
    if vmax is not None:
        _el(p, "valuemax", vmax)
    if isinstance(value, tuple):
        v = _el(p, "value")
        _el(v, "horiz", value[0])
        _el(v, "vert", value[1])
    else:
        _el(p, "value", value)


def _motion_filters(clip, pl, frame_w: int, frame_h: int):
    """Crop (percent) + Basic Motion (scale/center) matching the placement."""
    f1 = _el(clip, "filter")
    eff = _el(f1, "effect")
    _el(eff, "name", "Crop")
    _el(eff, "effectid", "crop")
    _el(eff, "effectcategory", "motion")
    _el(eff, "effecttype", "motion")
    _el(eff, "mediatype", "video")
    _param(eff, "left", "left", pl.crop_l, 0, 100)
    _param(eff, "right", "right", pl.crop_r, 0, 100)
    _param(eff, "top", "top", pl.crop_t, 0, 100)
    _param(eff, "bottom", "bottom", pl.crop_b, 0, 100)

    f2 = _el(clip, "filter")
    eff = _el(f2, "effect")
    _el(eff, "name", "Basic Motion")
    _el(eff, "effectid", "basic")
    _el(eff, "effectcategory", "motion")
    _el(eff, "effecttype", "motion")
    _el(eff, "mediatype", "video")
    _param(eff, "scale", "Scale", round(pl.scale * 100, 4), 0, 10000)
    _param(eff, "rotation", "Rotation", 0, -8640, 8640)
    _param(eff, "center", "Center", (round(pl.tx / frame_w, 6), round(pl.ty / frame_h, 6)))
    _param(eff, "centerOffset", "Anchor Point", (0, 0))


def write_fcp7_xml(
    out_path: str | Path,
    info: MediaInfo,
    inventory: dict,
    cuts: list[dict],
    project_name: str = "AI Narrative Rough Cut",
) -> Path:
    timebase = round(info.fps) if info.is_ntsc else max(1, round(info.fps))
    ntsc = info.is_ntsc
    to_f = lambda t: round(t * info.fps)  # noqa: E731
    total_frames = to_f(info.duration_s)
    src_name = Path(info.path).name
    file_id = "syncmaster-file-1"
    file_defined = False

    def file_el(parent):
        nonlocal file_defined
        if not file_defined:
            f = _el(parent, "file", id=file_id)
            _el(f, "name", src_name)
            _el(f, "pathurl", Path(info.path).as_uri())
            _rate(f, timebase, ntsc)
            _el(f, "duration", total_frames)
            media = _el(f, "media")
            video = _el(media, "video")
            sc = _el(video, "samplecharacteristics")
            _rate(sc, timebase, ntsc)
            _el(sc, "width", info.width)
            _el(sc, "height", info.height)
            _el(sc, "anamorphic", "FALSE")
            _el(sc, "pixelaspectratio", "square")
            if info.has_audio:
                audio = _el(media, "audio")
                asc = _el(audio, "samplecharacteristics")
                _el(asc, "depth", 16)
                _el(asc, "samplerate", info.audio_sample_rate or 48000)
                _el(audio, "channelcount", 2)
            file_defined = True
        else:
            _el(parent, "file", id=file_id)

    root = etree.Element("xmeml", version="4")
    seq = _el(root, "sequence", id="sequence-1")
    _el(seq, "name", project_name)
    seq_dur = to_f(cuts[-1]["end"]) if cuts else total_frames
    _el(seq, "duration", seq_dur)
    _rate(seq, timebase, ntsc)
    media = _el(seq, "media")
    video = _el(media, "video")
    vformat = _el(video, "format")
    sc = _el(vformat, "samplecharacteristics")
    _rate(sc, timebase, ntsc)
    _el(sc, "width", info.width)
    _el(sc, "height", info.height)
    _el(sc, "anamorphic", "FALSE")
    _el(sc, "pixelaspectratio", "square")
    _el(sc, "fielddominance", "none")
    track_v1 = _el(video, "track")
    track_v2 = _el(video, "track")  # SBS right-half layer

    clip_n = 0
    for i, cut in enumerate(cuts, 1):
        f0, f1 = to_f(cut["start"]), to_f(cut["end"])
        if f1 <= f0:
            continue
        placements = placements_for_cut(cut, inventory)
        for layer, (cam, pl) in enumerate(placements):
            track = track_v1 if layer == 0 else track_v2
            track.append(etree.Comment(cut_comment(cut, i)))
            clip_n += 1
            clip = _el(track, "clipitem", id=f"clipitem-{clip_n}")
            _el(clip, "name", f"{'+'.join(cut['camera_labels'])} [{cam}]")
            _el(clip, "enabled", "TRUE")
            _el(clip, "duration", f1 - f0)
            _rate(clip, timebase, ntsc)
            _el(clip, "start", f0)
            _el(clip, "end", f1)
            _el(clip, "in", f0)   # synced multicam: source time == timeline time
            _el(clip, "out", f1)
            file_el(clip)
            comments = _el(clip, "comments")
            _el(comments, "mastercomment1", f"{cut['rule']}: {cut['reason']}")
            if cut.get("markers"):
                _el(comments, "mastercomment2", "; ".join(m["comment"] for m in cut["markers"]))
            _motion_filters(clip, pl, info.width, info.height)

    # program audio: one continuous clip (editors re-cut audio themselves)
    if info.has_audio:
        audio = _el(media, "audio")
        _el(audio, "numOutputChannels", 2)
        atrack = _el(audio, "track")
        aclip = _el(atrack, "clipitem", id="clipitem-audio-1")
        _el(aclip, "name", src_name)
        _el(aclip, "enabled", "TRUE")
        _el(aclip, "duration", seq_dur)
        _rate(aclip, timebase, ntsc)
        _el(aclip, "start", 0)
        _el(aclip, "end", seq_dur)
        _el(aclip, "in", 0)
        _el(aclip, "out", seq_dur)
        file_el(aclip)
        st = _el(aclip, "sourcetrack")
        _el(st, "mediatype", "audio")
        _el(st, "trackindex", 1)

    # sequence markers for editor guidance
    for cut in cuts:
        for m in cut.get("markers", []):
            mk = _el(seq, "marker")
            _el(mk, "comment", m["comment"])
            _el(mk, "name", m["name"])
            _el(mk, "in", to_f(m["time"]))
            _el(mk, "out", -1)

    out_path = Path(out_path)
    out_path.write_bytes(
        etree.tostring(
            root, xml_declaration=True, encoding="UTF-8", pretty_print=True,
            doctype='<!DOCTYPE xmeml>',
        )
    )
    return out_path
