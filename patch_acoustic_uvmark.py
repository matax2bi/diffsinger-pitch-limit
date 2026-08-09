# -*- coding: utf-8 -*-
"""DiffSinger acoustic ONNX voicing mel-marker patch (3-way)
================================================================================
Stamps a voicing marker into the TOP mel bin (127), computed inside the
acoustic graph from (tokens, durations). Three classes per phoneme:
  +10.0  force-UNVOICED : special symbols (AP/SP/exh) + unvoiced-type consonants
  -20.0  force-VOICED   : vowels + always-voiced consonants (nasal/liquid/
                          semivowel) — protects them from voicing-detector errors
  (none) autonomous     : lenis-type consonants and untyped phonemes only —
                          the vocoder decides via its own mel-based voicing
                          detector (handles e.g. Korean lenis g/d/b/j, which are
                          unvoiced word-initially, voiced between vowels)
A marker-aware vocoder reads the marker as a deterministic F0 gate, then
restores the bin before rendering.

!!! WARNING !!!
Only use with a MARKER-AWARE vocoder. Standard vocoders (nsf_hifigan,
pc_nsf_hifigan, ...) render mel directly — the marker would produce a loud
16 kHz artifact on every marked frame.

Classification is driven by the dsdict.yaml `symbols:` types:
--special symbols (default AP,SP,exh) are always force-unvoiced (highest
priority, even if the dict labels them vowel); type=vowel and types in
--voiced-types (default nasal,liquid,semivowel,voiced) are force-voiced; types in
--unvoiced-types (default stop,fricative,affricate,aspirate,unvoiced) are
force-unvoiced; only remaining types (e.g. a custom 'lenis') and untyped
phonemes are autonomous. Relabel voiced-or-voiceless consonants (Korean
g/d/b/j, English b/d/g/...) to a type like 'lenis' in your dsdict to give
them to the vocoder's own detector.

Usage:
  python patch_acoustic_uvmark.py <vb>/dsmain/acoustic.onnx --dsdict <vb>/dsvariance/dsdict.yaml
  Options: --list-only, --special s1,s2, --unvoiced-types t1,t2,
           --voiced-types t1,t2, --unvoiced p1,p2, --voiced p1,p2, --out
"""
import os, re, sys, json, argparse
import numpy as np

MARK = 10.0


def load_types(dsdict_path):
    types = {}
    in_sym = False
    with open(dsdict_path, encoding='utf-8') as fh:
        for line in fh:
            if line.strip() == 'symbols:':
                in_sym = True
                continue
            if in_sym:
                m = re.match(r"\s*-\s*\{symbol:\s*([^,]+?),\s*type:\s*([^}]+?)\}", line)
                if m:
                    types[m.group(1).strip()] = m.group(2).strip()
                elif line.strip() and not line.startswith(' '):
                    break
    return types


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('acoustic')
    ap.add_argument('--dsdict', required=True)
    ap.add_argument('--phonemes', default=None,
                    help='phonemes.json (default: next to the acoustic onnx)')
    ap.add_argument('--unvoiced-types', default='stop,fricative,affricate,aspirate,unvoiced',
                    help='dsdict types treated as unvoiced (comma separated)')
    ap.add_argument('--voiced-types', default='nasal,liquid,semivowel,voiced',
                    help='consonant types always voiced, force-marked with vowels (comma)')
    ap.add_argument('--special', default='AP,SP,exh',
                    help='always force-unvoiced symbols, overrides dict type (comma)')
    ap.add_argument('--unvoiced', default='', help='extra phonemes to force-mark (comma)')
    ap.add_argument('--voiced', default='', help='phonemes to exclude from marking (comma)')
    ap.add_argument('--list-only', action='store_true')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    import onnx
    from onnx import helper, TensorProto

    ph_path = args.phonemes or os.path.join(os.path.dirname(args.acoustic), 'phonemes.json')
    with open(ph_path, encoding='utf-8') as fh:
        ph2id = json.load(fh)
    types = load_types(args.dsdict)
    print(f'{len(ph2id)} phonemes | dsdict types found: {sorted(set(types.values()))}')

    # 3-way classification: specials first, then voiced/unvoiced types;
    # autonomous = remaining types (e.g. lenis) + untyped only
    special = {s.strip() for s in args.special.split(',') if s.strip()} & set(ph2id)
    unv_types = {s.strip() for s in args.unvoiced_types.split(',') if s.strip()}
    voi_types = {s.strip() for s in args.voiced_types.split(',') if s.strip()} | {'vowel'}
    unv = {p for p in ph2id if types.get(p) in unv_types} | special
    vow = {p for p in ph2id if types.get(p) in voi_types} - special
    unv |= {s.strip() for s in args.unvoiced.split(',') if s.strip()}
    unv -= {s.strip() for s in args.voiced.split(',') if s.strip()}
    unv &= set(ph2id)
    vow -= unv
    auto = sorted(set(ph2id) - unv - vow)
    print(f'{len(unv)} phonemes force-UNVOICED:')
    for p in sorted(unv):
        print(f'  {p:12s} (type={types.get(p, "?")}, id={ph2id[p]})')
    print(f'{len(vow)} force-VOICED (vowels + voiced consonants), '
          f'{len(auto)} autonomous (vocoder decides):')
    for p in auto:
        print(f'  {p:12s} (type={types.get(p, "untyped")}, id={ph2id[p]})')
    if args.list_only:
        return

    V = max(ph2id.values()) + 1
    table = np.zeros(V, np.float32)          # +1=force-unvoiced, -1=force-voiced, 0=auto
    for p in unv:
        table[ph2id[p]] = 1.0
    for p in vow:
        table[ph2id[p]] = -1.0

    m = onnx.load(args.acoustic)
    g = m.graph
    if any(x.startswith('uvm_') for nd in g.node for x in nd.output):
        sys.exit('Already uvmark-patched - run on an unpatched file instead.')
    opset = max(op.version for op in m.opset_import if op.domain in ('', 'ai.onnx'))
    in_names = [i.name for i in g.input]
    assert 'tokens' in in_names and 'durations' in in_names, f'unexpected inputs: {in_names}'
    mel_out = g.output[0].name
    for nd in g.node:
        for k, o in enumerate(nd.output):
            if o == mel_out:
                nd.output[k] = 'uvm_mel_raw'
    P = 'uvm_'
    inits = [
        helper.make_tensor(P + 'table', TensorProto.FLOAT, [V], table.tolist()),
        helper.make_tensor(P + 'ax2', TensorProto.INT64, [1], [2]),
        helper.make_tensor(P + 'ax1', TensorProto.INT64, [1], [1]),
        helper.make_tensor(P + 'i0', TensorProto.INT64, [1], [0]),
        helper.make_tensor(P + 'i1', TensorProto.INT64, [1], [1]),
        helper.make_tensor(P + 'i127', TensorProto.INT64, [1], [127]),
        helper.make_tensor(P + 'i128', TensorProto.INT64, [1], [128]),
        helper.make_tensor(P + 'one_i', TensorProto.INT64, [], [1]),
        helper.make_tensor(P + 'zero_i', TensorProto.INT64, [], [0]),
        helper.make_tensor(P + 'mark', TensorProto.FLOAT, [], [MARK]),       # force-unvoiced +10
        helper.make_tensor(P + 'markv', TensorProto.FLOAT, [], [-20.0]),     # force-voiced -20
        helper.make_tensor(P + 'zerof', TensorProto.FLOAT, [], [0.0]),
        helper.make_tensor(P + 'onef', TensorProto.FLOAT, [], [1.0]),
    ]
    n = []
    mk = helper.make_node
    n.append(mk('Shape', ['uvm_mel_raw'], [P + 'shape']))
    n.append(mk('Gather', [P + 'shape', P + 'i1'], [P + 'T1'], axis=0))
    n.append(mk('Squeeze', [P + 'T1', P + 'i0'], [P + 'T']) if opset >= 13
             else mk('Squeeze', [P + 'T1'], [P + 'T'], axes=[0]))
    n.append(mk('Range', [P + 'zero_i', P + 'T', P + 'one_i'], [P + 'ar']))
    n.append(mk('Unsqueeze', [P + 'ar', P + 'i0'], [P + 'ar1']) if opset >= 13
             else mk('Unsqueeze', [P + 'ar'], [P + 'ar1'], axes=[0]))
    n.append(mk('CumSum', ['durations', P + 'ax1'], [P + 'cum']))
    n.append(mk('Unsqueeze', [P + 'ar1', P + 'ax2'], [P + 'arE']) if opset >= 13
             else mk('Unsqueeze', [P + 'ar1'], [P + 'arE'], axes=[2]))
    n.append(mk('Unsqueeze', [P + 'cum', P + 'ax1'], [P + 'cumE']) if opset >= 13
             else mk('Unsqueeze', [P + 'cum'], [P + 'cumE'], axes=[1]))
    n.append(mk('GreaterOrEqual', [P + 'arE', P + 'cumE'], [P + 'ge']))
    n.append(mk('Cast', [P + 'ge'], [P + 'gei'], to=TensorProto.INT64))
    n.append(mk('ReduceSum', [P + 'gei', P + 'ax2'], [P + 'idx0'], keepdims=0)
             if opset >= 13 else
             mk('ReduceSum', [P + 'gei'], [P + 'idx0'], axes=[2], keepdims=0))
    n.append(mk('Shape', ['durations'], [P + 'dsh']))
    n.append(mk('Gather', [P + 'dsh', P + 'i1'], [P + 'N1'], axis=0))
    n.append(mk('Squeeze', [P + 'N1', P + 'i0'], [P + 'N']) if opset >= 13
             else mk('Squeeze', [P + 'N1'], [P + 'N'], axes=[0]))
    n.append(mk('Sub', [P + 'N', P + 'one_i'], [P + 'Nm1']))
    n.append(mk('Min', [P + 'idx0', P + 'Nm1'], [P + 'idx']))
    n.append(mk('Gather', [P + 'table', 'tokens'], [P + 'tokflag'], axis=0))
    n.append(mk('GatherElements', [P + 'tokflag', P + 'idx'], [P + 'uvf'], axis=1))
    n.append(mk('Unsqueeze', [P + 'uvf', P + 'ax2'], [P + 'uv3']) if opset >= 13
             else mk('Unsqueeze', [P + 'uvf'], [P + 'uv3'], axes=[2]))
    # 3-way stamp: uvf in {+1 unvoiced, -1 voiced, 0 auto}
    #   u = clip(uvf,0,1), v = clip(-uvf,0,1) -> bin127 = mel*(1-u-v) + 10*u + (-20)*v
    n.append(mk('Slice', ['uvm_mel_raw', P + 'i0', P + 'i127', P + 'ax2'], [P + 'melA']))
    n.append(mk('Slice', ['uvm_mel_raw', P + 'i127', P + 'i128', P + 'ax2'], [P + 'melB']))
    n.append(mk('Clip', [P + 'uv3', P + 'zerof', P + 'onef'], [P + 'u3']))
    n.append(mk('Neg', [P + 'uv3'], [P + 'uv3n']))
    n.append(mk('Clip', [P + 'uv3n', P + 'zerof', P + 'onef'], [P + 'v3']))
    n.append(mk('Sub', [P + 'onef', P + 'u3'], [P + 'inv0']))
    n.append(mk('Sub', [P + 'inv0', P + 'v3'], [P + 'inv']))
    n.append(mk('Mul', [P + 'melB', P + 'inv'], [P + 'b0']))
    n.append(mk('Mul', [P + 'u3', P + 'mark'], [P + 'b1']))
    n.append(mk('Mul', [P + 'v3', P + 'markv'], [P + 'b2']))
    n.append(mk('Add', [P + 'b0', P + 'b1'], [P + 'b01']))
    n.append(mk('Add', [P + 'b01', P + 'b2'], [P + 'melBm']))
    n.append(mk('Concat', [P + 'melA', P + 'melBm'], [mel_out], axis=2))
    g.initializer.extend(inits)
    g.node.extend(n)
    onnx.checker.check_model(m)

    out = args.out or re.sub(r'\.onnx$', '', args.acoustic) + '.uvmark.onnx'
    onnx.save(m, out)
    print('saved:', out)
    print("-> set the 'acoustic:' entry of dsconfig.yaml to the file above.")
    print('WARNING: requires a marker-aware vocoder (see header).')


if __name__ == '__main__':
    main()
