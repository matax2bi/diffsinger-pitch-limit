# -*- coding: utf-8 -*-
"""DiffSinger acoustic ONNX pitch-limit patch
================================================================================
Patches a DiffSinger acoustic .onnx so that pitches OUTSIDE a chosen range
(above a high limit note and/or below a low limit note) are transparently
transposed back into range ONLY for the acoustic model (timbre), while a
pitch-controllable (PC) vocoder still renders the true pitch.

Result: notes outside the singer's trained range keep a healthy in-range timbre
instead of collapsing — no manual SHFC (tone shift) curve drawing, no octave cap.

Requires: Python 3.8+, `pip install onnx numpy`  (no torch needed)
MUST be used together with a pitch-controllable vocoder (e.g. pc_nsf_hifigan).

Usage:
  python patch_acoustic_limit.py <voicebank>/acoustic.onnx --limit-high D5
  python patch_acoustic_limit.py acoustic.onnx --limit-high D5 --limit-low G3
  python patch_acoustic_limit.py acoustic.onnx --limit-low A2
  -> creates acoustic.limit<...>.onnx. Point the 'acoustic:' entry of
     dsconfig.yaml to the new file. (--limit is an alias of --limit-high.)

Modes:
  transpose (default): shifts out-of-range regions by WHOLE SEMITONES, chosen
      from a ~250ms moving average of f0 — vibrato and pitch inflections are
      preserved. No octave cap.
  clamp: simple clipping to [low, high]. Vibrato outside the range flattens.
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
def build_pre_model(hi_hz, lo_hz, mode, win, opset, ir_version):
    import onnx
    from onnx import helper, TensorProto

    inits, nodes = [], []
    mk = helper.make_node

    def scalar(name, val):
        inits.append(helper.make_tensor(name, TensorProto.FLOAT, [], [float(val)]))
        return name

    def axes1(name):
        inits.append(helper.make_tensor(name, TensorProto.INT64, [1], [1]))
        return name

    def un(name_in, name_out):                   # [1,T] -> [1,1,T]
        if opset >= 13:
            nodes.append(mk('Unsqueeze', [name_in, axes1(name_out + '_ax')], [name_out]))
        else:
            nodes.append(mk('Unsqueeze', [name_in], [name_out], axes=[1]))

    def sq(name_in, name_out):                   # [1,1,T] -> [1,T]
        if opset >= 13:
            nodes.append(mk('Squeeze', [name_in, axes1(name_out + '_ax')], [name_out]))
        else:
            nodes.append(mk('Squeeze', [name_in], [name_out], axes=[1]))

    inp = helper.make_tensor_value_info('f0_user', TensorProto.FLOAT, [1, 'T'])
    outp = helper.make_tensor_value_info('f0_limited', TensorProto.FLOAT, [1, 'T'])
    scalar('zero', 0.0); scalar('one', 1.0)
    nodes.append(mk('Greater', ['f0_user', 'zero'], ['v_b']))
    nodes.append(mk('Cast', ['v_b'], ['v'], to=TensorProto.FLOAT))

    if mode == 'clamp':
        # clip to [lo, hi] on voiced frames only (unvoiced 0 must stay 0)
        scalar('lim_lo', lo_hz if lo_hz else 0.0)
        hi_in = scalar('lim_hi', hi_hz) if hi_hz else ''
        nodes.append(mk('Clip', ['f0_user', 'lim_lo', hi_in], ['clipped']))
        nodes.append(mk('Mul', ['clipped', 'v'], ['f0_limited']))
    else:                                        # transpose (whole semitones)
        pad = win // 2
        inits.append(helper.make_tensor('Wones', TensorProto.FLOAT, [1, 1, win],
                                        [1.0] * win))
        scalar('k12ln2', 12.0 / np.log(2.0))
        scalar('negln2_12', -np.log(2.0) / 12.0)
        scalar('posln2_12', np.log(2.0) / 12.0)
        nodes.append(mk('Mul', ['f0_user', 'v'], ['fv']))
        un('fv', 'fv3'); un('v', 'v3')
        nodes.append(mk('Conv', ['fv3', 'Wones'], ['num3'], pads=[pad, pad]))
        nodes.append(mk('Conv', ['v3', 'Wones'], ['den3'], pads=[pad, pad]))
        nodes.append(mk('Clip', ['den3', 'one', ''], ['den_c']))
        nodes.append(mk('Div', ['num3', 'den_c'], ['f0s3']))
        sq('f0s3', 'f0s')                        # voiced-masked moving average
        nodes.append(mk('Clip', ['f0s', 'one', ''], ['f0s_c']))
        factors = []
        if hi_hz:                                # shift DOWN when above high limit
            scalar('lim_hi', hi_hz)
            nodes.append(mk('Greater', ['f0s', 'lim_hi'], ['hi_b']))
            nodes.append(mk('Cast', ['hi_b'], ['hi_f'], to=TensorProto.FLOAT))
            nodes.append(mk('Mul', ['hi_f', 'v'], ['hi_over']))
            nodes.append(mk('Div', ['f0s_c', 'lim_hi'], ['hi_ratio']))
            nodes.append(mk('Log', ['hi_ratio'], ['hi_lg']))
            nodes.append(mk('Mul', ['hi_lg', 'k12ln2'], ['hi_semi']))
            nodes.append(mk('Ceil', ['hi_semi'], ['hi_n0']))
            nodes.append(mk('Clip', ['hi_n0', 'zero', ''], ['hi_n']))
            nodes.append(mk('Mul', ['hi_n', 'negln2_12'], ['hi_ex']))
            nodes.append(mk('Exp', ['hi_ex'], ['hi_shift']))     # 2^(-n/12)
            nodes.append(mk('Sub', ['hi_shift', 'one'], ['hi_sm1']))
            nodes.append(mk('Mul', ['hi_over', 'hi_sm1'], ['hi_t1']))
            nodes.append(mk('Add', ['hi_t1', 'one'], ['hi_factor']))
            factors.append('hi_factor')
        if lo_hz:                                # shift UP when below low limit
            scalar('lim_lo', lo_hz)
            nodes.append(mk('Less', ['f0s', 'lim_lo'], ['lo_b']))
            nodes.append(mk('Cast', ['lo_b'], ['lo_f'], to=TensorProto.FLOAT))
            nodes.append(mk('Mul', ['lo_f', 'v'], ['lo_over']))
            nodes.append(mk('Div', ['lim_lo', 'f0s_c'], ['lo_ratio']))
            nodes.append(mk('Log', ['lo_ratio'], ['lo_lg']))
            nodes.append(mk('Mul', ['lo_lg', 'k12ln2'], ['lo_semi']))
            nodes.append(mk('Ceil', ['lo_semi'], ['lo_n0']))
            nodes.append(mk('Clip', ['lo_n0', 'zero', ''], ['lo_n']))
            nodes.append(mk('Mul', ['lo_n', 'posln2_12'], ['lo_ex']))
            nodes.append(mk('Exp', ['lo_ex'], ['lo_shift']))     # 2^(+n/12)
            nodes.append(mk('Sub', ['lo_shift', 'one'], ['lo_sm1']))
            nodes.append(mk('Mul', ['lo_over', 'lo_sm1'], ['lo_t1']))
            nodes.append(mk('Add', ['lo_t1', 'one'], ['lo_factor']))
            factors.append('lo_factor')
        if len(factors) == 2:                    # regions disjoint -> safe to multiply
            nodes.append(mk('Mul', factors, ['factor']))
        else:
            nodes.append(mk('Identity', [factors[0]], ['factor']))
        nodes.append(mk('Mul', ['f0_user', 'factor'], ['f0_limited']))

    graph = helper.make_graph(nodes, 'f0_limit_pre', [inp], [outp], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', opset)])
    model.ir_version = ir_version
    import onnx as _o
    _o.checker.check_model(model)
    return model


def ref_transpose(f0, hi_hz, lo_hz, win):
    """Numpy reference of the transpose math (for the self-test)."""
    v = (f0 > 0).astype(np.float64)
    ker = np.ones(win)
    num = np.convolve(f0 * v, ker, 'same')
    den = np.maximum(np.convolve(v, ker, 'same'), 1.0)
    f0s = num / den
    f0s_c = np.maximum(f0s, 1.0)
    fac = np.ones_like(f0s)
    if hi_hz:
        over = ((f0s > hi_hz) & (v > 0)).astype(np.float64)
        n = np.maximum(np.ceil(12.0 * np.log(f0s_c / hi_hz) / np.log(2.0)), 0.0)
        fac *= over * np.exp(-n * np.log(2.0) / 12.0) + (1.0 - over)
    if lo_hz:
        under = ((f0s < lo_hz) & (v > 0)).astype(np.float64)
        n = np.maximum(np.ceil(12.0 * np.log(lo_hz / f0s_c) / np.log(2.0)), 0.0)
        fac *= under * np.exp(n * np.log(2.0) / 12.0) + (1.0 - under)
    return (f0 * fac).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('acoustic', help='path to the voicebank acoustic .onnx')
    ap.add_argument('--limit-high', '--limit', dest='limit_high', default=None,
                    help='high limit note (D5, F#4, or MIDI number)')
    ap.add_argument('--limit-low', dest='limit_low', default=None,
                    help='low limit note (G3, A2, or MIDI number)')
    ap.add_argument('--mode', choices=['transpose', 'clamp'], default='transpose')
    ap.add_argument('--win', type=int, default=21,
                    help='smoothing window in frames for transpose decision (default 21, ~250ms)')
    ap.add_argument('--f0_input', default='f0', help="name of the f0 graph input (default 'f0')")
    ap.add_argument('--out', default=None, help='output path (default: auto-named next to input)')
    ap.add_argument('--inplace', action='store_true',
                    help='replace the original file (a .bak backup is created)')
    args = ap.parse_args()
    import onnx
    from onnx import compose

    if not args.limit_high and not args.limit_low:
        sys.exit('specify --limit-high and/or --limit-low')
    hi_hz = lo_hz = None
    tag = ''
    if args.limit_high:
        t = tone_of(args.limit_high); hi_hz = hz_of(t)
        print(f'high limit: {args.limit_high} = tone {t} = {hi_hz:.1f}Hz')
        tag += args.limit_high
    if args.limit_low:
        t = tone_of(args.limit_low); lo_hz = hz_of(t)
        print(f'low  limit: {args.limit_low} = tone {t} = {lo_hz:.1f}Hz')
        tag += ('lo' + args.limit_low)
    if hi_hz and lo_hz and lo_hz >= hi_hz:
        sys.exit('low limit must be below high limit')
    print(f'mode={args.mode}')

    ac = onnx.load(args.acoustic)
    if any(x.startswith('f0limit_') for nd in ac.graph.node for x in list(nd.output)):
        sys.exit('Already patched (f0limit nodes found) - run on the original file instead.')
    in_names = [i.name for i in ac.graph.input]
    if args.f0_input not in in_names:
        sys.exit(f"input '{args.f0_input}' not found. Acoustic inputs: {in_names}\n"
                 f"-> specify it with --f0_input <name>.")
    opset = max(op.version for op in ac.opset_import if op.domain in ('', 'ai.onnx'))

    pre = build_pre_model(hi_hz, lo_hz, args.mode, args.win, opset, ac.ir_version)
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
        out = args.out or re.sub(r'\.onnx$', '', args.acoustic) + f'.limit{tag}.onnx'
    onnx.save(merged, out)
    print('saved:', out)
    if not args.inplace and not args.out:
        print("-> set the 'acoustic:' entry of dsconfig.yaml to the file above.")
    print('NOTE: use together with a pitch-controllable (PC) vocoder.')

    # Self-test (if onnxruntime is installed)
    try:
        import onnxruntime as ort
    except ImportError:
        print('(onnxruntime not installed - numeric self-test skipped)')
        return
    if args.mode == 'transpose':
        sess = ort.InferenceSession(pre.SerializeToString(), providers=['CPUExecutionProvider'])
        mid = hi_hz / 2 ** 0.5 if hi_hz else lo_hz * 2 ** 0.5
        f0 = np.zeros((1, 260), np.float32)
        f0[0, 20:80] = (hi_hz * 2 ** (3 / 12)) if hi_hz else mid   # high+3st (or in-range)
        f0[0, 100:160] = mid                                        # in range
        f0[0, 180:240] = (lo_hz * 2 ** (-3 / 12)) if lo_hz else mid  # low-3st (or in-range)
        got = sess.run(None, {'f0_user': f0})[0][0]
        ref = ref_transpose(f0[0], hi_hz, lo_hz, args.win)
        match = (np.abs(got - ref) < 1.0).mean()
        oks = [f'reference match {match*100:.1f}% {"OK" if match > 0.95 else "FAIL"}']
        if hi_hz:
            hv = got[40:60].mean()
            oks.append(f'high+3st -> {hv:.1f}Hz (<= {hi_hz:.1f}) '
                       f'{"OK" if hv <= hi_hz * 1.001 else "FAIL"}')
        if lo_hz:
            lv = got[200:220].mean()
            oks.append(f'low-3st -> {lv:.1f}Hz (>= {lo_hz:.1f}) '
                       f'{"OK" if lv >= lo_hz * 0.999 else "FAIL"}')
        oks.append(f'in-range preserved {"OK" if abs(got[120:140].mean() - mid) < 0.5 else "FAIL"}')
        print('self-test:', ' | '.join(oks))


if __name__ == '__main__':
    main()
