# DiffSinger Acoustic Pitch-Limit Patch

**English** | [한국어](README.ko.md)

Fixes timbre collapse on notes **above a voicebank's trained vocal range** by
grafting a small "limit note" preprocessor into the acoustic ONNX file.

- Regions above the limit note are lowered **only for the acoustic model**, so the
  timbre is always generated inside the trained range
- The actual pitch is still rendered correctly by a **pitch-controllable (PC) vocoder**
- Same effect as hand-drawing an SHFC (tone shift) curve in OpenUtau, but
  **automatic** and with **no octave cap**

> ⚠️ **Must be used together with a PC vocoder** (e.g. `pc_nsf_hifigan`).
> With a regular vocoder the pitch itself would be lowered.

## Install

Python 3.8+ and one line:

```
pip install onnx numpy
```

(Optionally `pip install onnxruntime` to see the numeric self-test — patching works without it.)

## Usage

```
python patch_acoustic_limit.py <voicebank>/acoustic.onnx --limit D5
```

- This creates `acoustic.limitD5.onnx`.
- Open the voicebank's `dsconfig.yaml` and point the `acoustic:` entry to the new file.
- To revert: point `acoustic:` back to the original file (the original is never modified).
- On Windows you can also **drag & drop** the acoustic.onnx onto `drag_drop_patch.bat`.

Note names: `D5`, `F#4`, `Bb3`, or a MIDI number (`74`). C4 = 60.

### Options

| Option | Description |
|---|---|
| `--mode transpose` (default) | Shifts over-limit regions down by **whole semitones** — vibrato and pitch inflections are preserved. No octave cap |
| `--mode clamp` | Simple `min(f0, limit)` — vibrato above the limit gets flattened |
| `--inplace` | Replace the original file (a `.bak` backup is created) |
| `--f0_input <name>` | For models whose f0 input is not named `f0` |
| `--win N` | Moving-average window for the transpose decision (default 21 frames ≈ 250 ms) |

## How it works

In a pitch-controllable vocoder setup, **timbre follows the f0 the acoustic model
receives**, while **actual pitch follows the f0 the vocoder receives**. This tool
merges a tiny f0-preprocessing graph (~20 nodes) in front of the acoustic ONNX
using `onnx.compose`. Model weights are untouched and the added compute is
negligible. Pitches above your limit note are transposed down for the acoustic
only, so out-of-range notes keep a healthy in-range timbre while the PC vocoder
renders the true pitch.

## Notes

- Do not combine with OpenUtau's SHFC (tone shift) curve on the same notes —
  the shifts would stack. Use one or the other.
- Only the acoustic model is patched; variance/pitch models are untouched.
- If anything goes wrong, reverting `dsconfig.yaml` restores everything instantly.
