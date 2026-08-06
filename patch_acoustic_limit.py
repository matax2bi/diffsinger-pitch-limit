# -*- coding: utf-8 -*-
"""DiffSinger acoustic ONNX pitch-limit patch
================================================================================
Patches a DiffSinger acoustic .onnx so that pitches above a chosen "limit note"
are transparently lowered ONLY for the acoustic model (timbre), while a
pitch-controllable (PC) vocoder still renders the true pitch.

Result: notes above the singer's trained range keep a healthy in-range timbre
instead of collapsing — no manual SHFC (tone shift) curve drawing, no octave cap.

Requires: Python 3.8+, `pip install onnx numpy`  (no torch needed)
MUST be used together with a pitch-controllable vocoder (e.g. pc_nsf_hifigan).

Usage:
  python patch_acoustic_limit.py <voicebank>/acoustic.onnx --limit D5
  -> creates acoustic.limitD5.onnx. Point the 'acoustic:' entry of dsconfig.yaml
     to the new file. (Or use --inplace to replace the original; a .bak backup
     is created.)

Modes:
  transpose (default): shifts over-limit regions down by WHOLE SEMITONES, chosen
      from a ~250ms moving average of f0 — vibrato and pitch inflections are
      preserved. No octave cap.
  clamp: simple min(f0, limit). Vibrato above the limit gets flattened.
"""
import argparse, re, shutil, sys
import numpy as np

_NOTE = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}


def tone_of(s):
    """'D5' / 'F#4' / 'Bb3' / '74' -> MIDI tone number (C4=60)."""
    s = str(s).strip()
    if re.fullmatch(r'\d+', s):
        return int(s)
    m = re.fullmatch(r'([A-Ga-g])([#b]?)(-?\d+)', s)
    if not m:
        raise ValueError(f'cannot parse note name: {s}')
    t = _NOTE[m.group(1).upper()] + (1 if m.group(2) == '#' else -1 if m.group(2) == 'b' else 0)
    return t + (int(m.group(3)) + 1) * 12


def hz_of(tone):
    return 440.0 * 2.0 ** ((tone - 69) / 12.0)


# ------------------------------------------------------------------
# Build the f0-preprocessing subgraph directly with onnx.helper (no torch)
# ------------------------------------------------------------------
def build_pre_model(limit_hz, mode, win, opset, ir_version):
    import onnx
    from onnx import helper, TensorProto

    inits, nodes = [], []

    def scalar(name, val):
        inits.append(helper.make_tensor(name, TensorProto.FLOAT, [], [float(val)]))
        return name

    def axes1(name):   # axes [1] for Unsqueeze/Squeeze — input tensor since opset 13
        inits.append(helper.make_tensor(name, TensorProto.INT64, [1], [1]))
        return name

    inp = helper.make_tensor_value_info('f0_user', TensorProto.FLOAT, [1, 'T'])
    outp = helper.make_tensor_value_info('f0_limited', TensorProto.FLOAT, [1, 'T'])

    if mode == 'clamp':
        scalar('lim', limit_hz)
        nodes.append(helper.make_node('Clip', ['f0_user', '', 'lim'], ['f0_limited'],
                                      name='f0limit_clip'))
    else:                                            # transpose
        pad = win // 2
        inits.append(helper.make_tensor('Wones', TensorProto.FLOAT, [1, 1, win],
                                        [1.0] * win))
        scalar('zero', 0.0); scalar('one', 1.0); scalar('lim', limit_hz)
        scalar('k12ln2', 12.0 / np.log(2.0))         # log -> semitones
        scalar('negln2_12', -np.log(2.0) / 12.0)     # semitones -> ratio exponent

        def un(name_in, name_out):                   # [1,T] -> [1,1,T]
            if opset >= 13:
                ax = axes1(name_out + '_ax')
                nodes.append(helper.make_node('Unsqueeze', [name_in, ax], [name_out]))
            else:
                nodes.append(helper.make_node('Unsqueeze', [name_in], [name_out], axes=[1]))

        def sq(name_in, name_out):                   # [1,1,T] -> [1,T]
            if opset >= 13:
                ax = axes1(name_out + '_ax')
                nodes.append(helper.make_node('Squeeze', [name_in, ax], [name_out]))
            else:
                nodes.append(helper.make_node('Squeeze', [name_in], [name_out], axes=[1]))

        n = nodes.append
        n(helper.make_node('Greater', ['f0_user', 'zero'], ['v_b']))
        n(helper.make_node('Cast', ['v_b'], ['v'], to=TensorProto.FLOAT))
        n(helper.make_node('Mul', ['f0_user', 'v'], ['fv']))
        un('fv', 'fv3'); un('v', 'v3')
        n(helper.make_node('Conv', ['fv3', 'Wones'], ['num3'], pads=[pad, pad]))
        n(helper.make_node('Conv', ['v3', 'Wones'], ['den3'], pads=[pad, pad]))
        n(helper.make_node('Clip', ['den3', 'one', ''], ['den_c']))
        n(helper.make_node('Div', ['num3', 'den_c'], ['f0s3']))
        sq('f0s3', 'f0s')                            # voiced-masked moving average (removes vibrato)
        n(helper.make_node('Greater', ['f0s', 'lim'], ['over_b']))
        n(helper.make_node('Cast', ['over_b'], ['over_f'], to=TensorProto.FLOAT))
        n(helper.make_node('Mul', ['over_f', 'v'], ['over']))
        n(helper.make_node('Clip', ['f0s', 'one', ''], ['f0s_c']))
        n(helper.make_node('Div', ['f0s_c', 'lim'], ['ratio']))
        n(helper.make_node('Log', ['ratio'], ['lg']))
        n(helper.make_node('Mul', ['lg', 'k12ln2'], ['semi']))
        n(helper.make_node('Ceil', ['semi'], ['n0']))
        n(helper.make_node('Clip', ['n0', 'zero', ''], ['nsemi']))
        n(helper.make_node('Mul', ['nsemi', 'negln2_12'], ['ex']))
        n(helper.make_node('Exp', ['ex'], ['shift']))          # 2^(-n/12)
        n(helper.make_node('Sub', ['shift', 'one'], ['shift_m1']))
        n(helper.make_node('Mul', ['over', 'shift_m1'], ['t1']))
        n(helper.make_node('Add', ['t1', 'one'], ['factor']))  # over ? shift : 1
        n(helper.make_node('Mul', ['f0_user', 'factor'], ['f0_limited']))

    graph = helper.make_graph(nodes, 'f0_limit_pre', [inp], [outp], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', opset)])
    model.ir_version = ir_version
    import onnx as _o
    _o.checker.check_model(model)
    return model


def ref_transpose(f0, limit_hz, win):
    """Numpy reference of the transpose math (for the self-test)."""
    v = (f0 > 0).astype(np.float64)
    ker = np.ones(win)
    num = np.convolve(f0 * v, ker, 'same')
    den = np.maximum(np.convolve(v, ker, 'same'), 1.0)
    f0s = num / den
    over = ((f0s > limit_hz) & (v > 0)).astype(np.float64)
    nsemi = np.maximum(np.ceil(12.0 * np.log(np.maximum(f0s, 1.0) / limit_hz) / np.log(2.0)), 0.0)
    shift = np.exp(-nsemi * np.log(2.0) / 12.0)
    return (f0 * (over * shift + (1.0 - over))).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('acoustic', help='path to the voicebank acoustic .onnx')
    ap.add_argument('--limit', required=True, help='limit note (D5, F#4, Bb3, or MIDI number)')
    ap.add_argument('--mode', choices=['transpose', 'clamp'], default='transpose')
    ap.add_argument('--win', type=int, default=21,
                    help='smoothing window in frames for transpose decision (default 21, ~250ms)')
    ap.add_argument('--f0_input', default='f0', help="name of the f0 graph input (default 'f0')")
    ap.add_argument('--inplace', action='store_true',
                    help='replace the original file (a .bak backup is created)')
    args = ap.parse_args()
    import onnx
    from onnx import compose

    tone = tone_of(args.limit)
    limit_hz = hz_of(tone)
    print(f'limit note: {args.limit} = tone {tone} = {limit_hz:.1f}Hz, mode={args.mode}')

    ac = onnx.load(args.acoustic)
    in_names = [i.name for i in ac.graph.input]
    if args.f0_input not in in_names:
        sys.exit(f"input '{args.f0_input}' not found. Acoustic inputs: {in_names}\n"
                 f"-> specify it with --f0_input <name>.")
    opset = max(op.version for op in ac.opset_import if op.domain in ('', 'ai.onnx'))

    pre = build_pre_model(limit_hz, args.mode, args.win, opset, ac.ir_version)
    merged = compose.merge_models(pre, ac, io_map=[('f0_limited', args.f0_input)],
                                  prefix1='f0limit_')
    for i in merged.graph.input:                     # restore the original input name 'f0'
        if i.name == 'f0limit_f0_user':
            i.name = args.f0_input
    for nd in merged.graph.node:
        for k, x in enumerate(nd.input):
            if x == 'f0limit_f0_user':
                nd.input[k] = args.f0_input
    onnx.checker.check_model(merged)

    if args.inplace:
        shutil.copy2(args.acoustic, args.acoustic + '.bak')
        out = args.acoustic
    else:
        out = re.sub(r'\.onnx$', '', args.acoustic) + f'.limit{args.limit}.onnx'
    onnx.save(merged, out)
    print('saved:', out)
    if not args.inplace:
        print("-> set the 'acoustic:' entry of dsconfig.yaml to the file above.")
    print('NOTE: use together with a pitch-controllable (PC) vocoder.')

    # Self-test (if onnxruntime is installed): run the pre-graph alone vs numpy reference
    try:
        import onnxruntime as ort
    except ImportError:
        print('(onnxruntime not installed - numeric self-test skipped)')
        return
    if args.mode == 'transpose':
        sess = ort.InferenceSession(pre.SerializeToString(), providers=['CPUExecutionProvider'])
        f0 = np.zeros((1, 200), np.float32)
        f0[0, 20:80] = hz_of(tone + 3)               # limit + 3 semitones
        f0[0, 100:160] = hz_of(tone - 2)             # within range
        got = sess.run(None, {'f0_user': f0})[0][0]
        ref = ref_transpose(f0[0], limit_hz, args.win)
        # frames at exact ceil boundaries may flip one semitone (fp32 vs fp64) -> use match rate
        match = (np.abs(got - ref) < 1.0).mean()
        hi = got[40:60].mean()
        ok1 = match > 0.95
        ok2 = hi <= limit_hz * 1.001
        ok3 = abs(got[120:140].mean() - hz_of(tone - 2)) < 0.5
        print(f'self-test: reference match {match*100:.1f}% {"OK" if ok1 else "FAIL"} | '
              f'limit+3st -> {hi:.1f}Hz (<= {limit_hz:.1f}) {"OK" if ok2 else "FAIL"} | '
              f'in-range preserved {"OK" if ok3 else "FAIL"}')


if __name__ == '__main__':
    main()
