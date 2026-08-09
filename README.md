# DiffSinger Acoustic Pitch-Limit Patch

**English** | [한국어](README.ko.md)

<<<<<<< HEAD
Fixes timbre collapse on notes **outside a voicebank's trained vocal range —
both above AND below** — by grafting a small "limit note" preprocessor into
the acoustic ONNX file.
=======


https://github.com/user-attachments/assets/0ac279cf-2716-45b8-bb12-f3ded3a8219c

(Heat abnormal / Iyowa feat.Adachi Rei / UST. @music_is_genre)


Fixes timbre collapse on notes **above a voicebank's trained vocal range** by
grafting a small "limit note" preprocessor into the acoustic ONNX file.
>>>>>>> 22f26512d25ce07e5a0613409ad3abef284efb5d

- Regions above the high limit are lowered, and regions below the low limit
  are raised, **only for the acoustic model** — so the timbre is always
  generated inside the trained range
- The actual pitch is still rendered correctly by a **pitch-controllable (PC) vocoder**
- Same effect as hand-drawing an SHFC (tone shift) curve in OpenUtau, but
  **automatic** and with **no octave cap**

> ⚠️ **Must be used together with a PC vocoder** — e.g.
> [ezv_for_diffsinger](https://github.com/matax2bi/ezv-for-diffsinger)
> (made to pair with this tool) or `pc_nsf_hifigan`.
> With a regular vocoder the pitch itself would shift.

## Install

Python 3.8+ and one line:

```
pip install onnx numpy
```

(Optionally `pip install onnxruntime` to see the numeric self-test — patching works without it.)

## Usage 1 — drag & drop (Windows, recommended)

1. Drag your voicebank's `acoustic.onnx` onto **`drag_drop_patch.bat`**.
2. Answer the prompts: high limit note → low limit note (press Enter to
   skip either; note names like `D5`, `F#4`, `Bb3` or MIDI numbers) →
   uv-mark yes/no (`y` only if you use the
   [ezv_for_diffsinger](https://github.com/matax2bi/ezv-for-diffsinger)
   vocoder).
3. `acoustic.patched.onnx` is created next to the original.
4. Open the voicebank's `dsconfig.yaml` and point the `acoustic:` entry to
   the new file.
5. To revert: point `acoustic:` back to the original file (the original is
   never modified).

## Usage 2 — command line

```
python patch_acoustic_limit.py <voicebank>/acoustic.onnx --limit-high D5
python patch_acoustic_limit.py <voicebank>/acoustic.onnx --limit-high D5 --limit-low G3
python patch_acoustic_limit.py <voicebank>/acoustic.onnx --limit-low A2
```

- This creates `acoustic.limit<...>.onnx` (high / low / both can be set).
- Point `dsconfig.yaml`'s `acoustic:` entry to the new file, same as above.

Note names: `D5`, `F#4`, `Bb3`, or a MIDI number (`74`). C4 = 60.

### Options

| Option | Description |
|---|---|
| `--limit-high NOTE` (alias `--limit`) | High limit — notes above are shifted down |
| `--limit-low NOTE` | Low limit — notes below are shifted **up** (can combine with high) |
| `--mode transpose` (default) | Shifts out-of-range regions by **whole semitones** — vibrato and pitch inflections are preserved. No octave cap |
| `--mode clamp` | Simple clipping to `[low, high]` — vibrato outside the range flattens |
| `--inplace` | Replace the original file (a `.bak` backup is created) |
| `--f0_input <name>` | For models whose f0 input is not named `f0` |
| `--win N` | Moving-average window for the transpose decision (default 21 frames ≈ 250 ms) |
| `--out PATH` | Explicit output path |

### Bonus tool: voicing mel-marker (`patch_acoustic_uvmark.py`)

**This feature is made for the
[ezv_for_diffsinger](https://github.com/matax2bi/ezv-for-diffsinger) vocoder**
— it is the only vocoder that reads these markers; other vocoders will not
understand them.

Stamps a 3-way voicing marker into the top mel bin, decided per phoneme from
your dsdict.yaml `symbols:` types: **force-unvoiced** (AP/SP/exh specials +
unvoiced-type consonants), **force-voiced** (vowels + always-voiced consonants),
and **autonomous** (a marker-aware vocoder decides per frame — handles phonemes
that can be either, like Korean lenis g/d/b/j). Set your dsdict types like:

- voiceless consonants → `stop` / `fricative` / `affricate` / `aspirate`
- always-voiced consonants (nasals, liquids, ...) → `nasal` / `liquid` / `voiced`
- position-dependent voicing (e.g. Korean lenis g/d/b/j) → `lenis`

⚠️ **Do NOT render a uv-marked acoustic with standard vocoders**
(nsf_hifigan / pc_nsf_hifigan render mel directly — the marker becomes a loud
16 kHz beep). **Back up your original voicebank before patching**; to go back
to another vocoder, point dsconfig's acoustic back to the original file. The
drag & drop bat asks about this step and defaults to No.

Note: the voicing table is **baked into the onnx at patch time**. If you edit
your dsdict types later, re-run the patch (on the un-marked file) — the already
patched onnx does not follow dictionary edits.

## How it works

In a pitch-controllable vocoder setup, **timbre follows the f0 the acoustic model
receives**, while **actual pitch follows the f0 the vocoder receives**. This tool
merges a tiny f0-preprocessing graph (~20 nodes) in front of the acoustic ONNX
using `onnx.compose`. Model weights are untouched and the added compute is
negligible. Out-of-range pitches are transposed back into range (down past the
high limit, up past the low limit) for the acoustic only, so out-of-range notes
keep a healthy in-range timbre while the PC vocoder renders the true pitch.

## Notes

- Do not combine with OpenUtau's SHFC (tone shift) curve on the same notes —
  the shifts would stack. Use one or the other.
- Only the acoustic model is patched; variance/pitch models are untouched.
- If anything goes wrong, reverting `dsconfig.yaml` restores everything instantly.
