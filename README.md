# AI Narrative Video Director

Turns a multicam podcast **SyncMaster** recording into a production-ready
rough cut: an FCPXML v1.10 + a Premiere-native FCP7 XML that import as an
already-edited sequence, with every cut annotated for the human editor.

## Quick start

```bash
pip install -r requirements.txt          # needs ffmpeg on PATH (brew install ffmpeg)
cp .env.example .env                     # add your API key (OpenRouter by default)
python run.py path/to/syncmaster.mp4     # full run with human checkpoints
```

Put `show_type.txt` (containing "The Nav Thethi Show" or
"Cracking the Maturity Code") next to the video or in the project root.

Flags: `--auto` (skip human checkpoints) · `--force` (ignore stage caches).

Outputs land in `output/`:
- `output.fcpxml` — FCPXML v1.10 (the brief's literal spec)
- `output_premiere.xml` — FCP7 xmeml, Premiere's native XML interchange
- `editing_report.json` — camera inventory, speaker map, every cut with its rule
  and reason, warnings, off-camera segments, validation results

## Architecture

```
SyncMaster.mp4
   │ ffprobe / ffmpeg / OpenCV
   ▼
[1] Camera Discovery      grid geometry (no hardcoded coords) + vision LLM roles
   ▼                      ── HITL checkpoint: confirm camera mapping ──
[2] Transcription         faster-whisper, WORD-level timestamps
    Speaker Mapping       per-tile mouth-motion vs speech (vision-correlated
                          diarization) + LLM host/guest from the opening
   ▼                      ── HITL checkpoint: confirm host/guest ──
[3] Narrative Analysis    LLM: questions/stories/emotional beats/laughter/topics
    Visual Events         vision LLM: phys adjustments + reactions (budgeted);
                          deterministic: freezes, "stop rolling" phrase scan
   ▼
[4] Decision Engine       DETERMINISTIC rules engine — all 10 editorial rules
                          + show profiles. No LLM: same inputs => same cuts.
   ▼                      ── HITL checkpoint: review/edit cut list ──
[5] XML Generators        FCPXML 1.10 + FCP7 xmeml (shared crop/transform math)
   ▼
[6] Validation            XML parse, contiguous timeline, mid-word-cut audit,
                          mandatory-marker coverage → editing_report.json
```

Stages exchange plain JSON and are cached per-video in `.cache/` — re-runs
never re-pay transcription or vision-API costs. Delete the cache or use
`--force` to recompute.

### Key design decisions

- **Vision-correlated diarization instead of pyannote.** Audio diarization
  can say "Speaker A" but never "Speaker A is on cam_3". Since per-tile mouth
  activity must be measured anyway for that mapping, the visual signal *is*
  the diarizer (one cheap ffmpeg decode pass). The diarizer sits behind a
  clean interface, so pyannote can be slotted in for crosstalk-heavy shows.
- **LLM-free decision engine.** Upstream AI produces *facts*; the editorial
  rulebook is deterministic, testable code. Deterministic output was a hard
  requirement, and "why did it cut here?" always has an exact answer —
  every cut carries `rule` + `reason` in the report and in XML comments.
- **Rule 9 is regex, not LLM.** "Stop rolling" is a production-control
  directive; it is detected deterministically on word timestamps.
- **Marker preservation.** When a higher-priority rule (freeze, off-camera)
  overrides a shot, markers like `PHY_ADJ_CUT` migrate to the replacement
  shot — mandatory annotations can never silently vanish.
- **Cost guard.** All vision-LLM sampling shares `analysis.max_vision_calls`;
  long episodes degrade gracefully (with a warning) instead of running up an
  unbounded bill.
- **One source asset.** Every clip references the SyncMaster file with exact
  crop + uniform scale + reposition (shared math in `stages/xml_common.py`),
  so each cut shows one camera tile full-frame — or two tiles for SBS
  split-screens on "Cracking the Maturity Code".

### Editorial rules → code map (`stages/decision_engine.py`)

| Rule | Implementation |
|---|---|
| 1 Speaker | base timeline on current speaker's hero cam |
| 2 Listener reaction | vision reaction events → 3–5s cutaways, rate-limited, blocked during emotional holds |
| 3 Refresh | 45s without events → 3s wide |
| 4 Dialogue | ≥4 alternating turns <4s → wide (Nav) / SBS (Maturity) |
| 5 Monologue | >30s single turn → alternate angle every ~25s |
| 6 Emotional priority | high-intensity story/emotion spans block all optional cuts |
| 7 Physical adjustment | mandatory locked cutaway + `PHY_ADJ_CUT` marker/comment |
| 8 Technical failure | freeze detection → locked switch + `TECH_FAILURE` marker |
| 9 Off-camera | "stop/restart rolling" → locked `OFF_CAMERA_BRAINSTORM` segment |
| 10 Safety | boundaries snapped to word-gap centers, out of laughter; ≥2s min shot |

Show profiles: **Nav Thethi** — hero-first pauses, wide budget enforced <20%
of publishable runtime. **Maturity Code** — intro/opening question/outro and
shared laughter in SBS split-screen.

## Human-in-the-loop

Three gates (configurable in `config.yaml → hitl`):
1. **cameras** — table of detected tiles/roles; override with `CAM_WIDE=cam_5`
   or `cam_2:EMPTY`.
2. **speakers** — host/guest mapping with sample lines; `swap` or `host=cam_X`.
3. **cuts** — full cut list written to `output/cuts_preview.json`; edit the
   file, type `reload`, and the edited list is what gets rendered to XML.

Every override is recorded in `editing_report.json → metadata.hitl_overrides`.

## Configuration

Everything lives in `config.yaml` — provider (`openrouter` / `anthropic` /
`openai` / `custom` base URL), model ids, whisper size, sampling rates,
vision budget, every editorial threshold. API keys come only from env/.env
(never from the config file).

## Testing

```bash
python scripts/make_synthetic.py   # builds a 3x2 SyncMaster with `say` dialogue
python scripts/test_e2e.py         # full pipeline with a mock LLM + assertions
```

The synthetic episode includes a scripted "stop rolling … restart rolling"
span, an emotional story beat, a physical adjustment, and empty camera
tiles — asserting grid detection, speaker mapping, Rule 7/9 markers, the
Nav Thethi wide budget, and XML validity.

## Known caveats (state honestly at review)

- FCPXML transform `position` conventions vary between NLE importers; the
  Premiere-native FCP7 XML is the guaranteed-import path. If tiles appear
  offset after import, the convention constant lives in one place
  (`xml_fcpxml.py` / `xml_fcp7.py` position lines).
- Locked mandatory cuts (Rule 7/9) may land mid-word by design — the rules
  demand immediate cutaways; validation reports them as warnings.
- Reaction detection quality is bounded by `analysis.vision_interval_s`;
  denser sampling costs more vision calls.
