# DiffSinger 어쿠스틱 한계음 패치

[English](README.md) | **한국어**

DiffSinger 보이스뱅크의 **학습 음역 밖 — 고음과 저음 모두 —** 에서 음색이
무너지는 문제를, 어쿠스틱 ONNX 파일에 "한계음" 전처리를 이식해서 해결하는
도구입니다.

- 상한을 넘는 구간은 낮춰서, 하한을 밑도는 구간은 올려서 **어쿠스틱에게만**
  들려줌 — 음색은 항상 학습 범위 안에서 생성
- 실제 음정은 **PC(pitch controllable) 보코더**가 원래 f0 그대로 렌더
- OpenUtau에서 SHFC(톤 시프트) 커브를 손으로 그리는 것과 같은 효과를 **자동으로**, 옥타브 제한 없이

> ⚠️ **반드시 PC 보코더와 함께 사용하세요** — 예:
> [ezv_for_diffsinger](https://github.com/matax2bi/ezv-for-diffsinger)
> (이 도구와 짝으로 만든 보코더) 또는 `pc_nsf_hifigan`.
> 일반 보코더로 렌더하면 음정 자체가 이동해 버립니다.

## 설치

Python 3.8 이상 + 아래 한 줄:

```
pip install onnx numpy
```

(수치 self-test까지 보려면 `pip install onnxruntime` 추가 — 없어도 패치는 됩니다.)

## 사용법 1 — 드래그&드롭 (Windows, 추천)

1. 보이스뱅크의 `acoustic.onnx` 를 **`drag_drop_patch.bat`** 에 끌어다 놓습니다.
2. 물음에 답하세요: 상한 음 → 하한 음 (건너뛰려면 그냥 Enter — 표기는
   `D5`, `F#4`, `Bb3` 또는 MIDI 번호) → uv 마크 적용 여부
   ([ezv_for_diffsinger](https://github.com/matax2bi/ezv-for-diffsinger)
   보코더를 쓸 때만 `y`).
3. 원본 옆에 `acoustic.patched.onnx` 가 생성됩니다.
4. 보이스뱅크의 `dsconfig.yaml` 을 열어 `acoustic:` 항목을 새 파일명으로
   바꾸면 끝.
5. 되돌리기: `acoustic:` 항목을 원래 파일명으로 복원 (원본 파일은 수정되지
   않습니다).

## 사용법 2 — 명령줄

```
python patch_acoustic_limit.py <보이스뱅크 폴더>/acoustic.onnx --limit-high D5
python patch_acoustic_limit.py <보이스뱅크 폴더>/acoustic.onnx --limit-high D5 --limit-low G3
python patch_acoustic_limit.py <보이스뱅크 폴더>/acoustic.onnx --limit-low A2
```

- `acoustic.limit<...>.onnx` 가 생성됩니다 (상한/하한/둘 다 지정 가능).
- 위와 마찬가지로 `dsconfig.yaml` 의 `acoustic:` 항목을 새 파일명으로 변경.

한계음 표기: `D5`, `F#4`, `Bb3`, 또는 MIDI 번호(`74`). C4 = 60 기준.

### 옵션

| 옵션 | 설명 |
|---|---|
| `--limit-high 음` (별칭 `--limit`) | 상한 — 넘는 음을 반음 단위로 내림 |
| `--limit-low 음` | 하한 — 밑도는 음을 반음 단위로 **올림** (상한과 동시 사용 가능) |
| `--mode transpose` (기본) | 범위 밖 구간을 **반음 단위로 통째로 이동** — 비브라토·피치 굴곡 보존. 옥타브 제한 없음 |
| `--mode clamp` | 단순 `[하한, 상한]` 클리핑 — 범위 밖 비브라토가 평탄해짐 |
| `--inplace` | 원본 파일을 직접 교체 (원본은 `.bak` 으로 백업) |
| `--f0_input <이름>` | f0 입력 이름이 `f0` 가 아닌 모델용 |
| `--win N` | transpose 판단용 이동평균 길이 (기본 21프레임 ≈ 250ms) |
| `--out 경로` | 출력 경로 직접 지정 |

### 보너스 도구: 유/무성 mel 마커 (`patch_acoustic_uvmark.py`)

**이 기능은
[ezv_for_diffsinger](https://github.com/matax2bi/ezv-for-diffsinger)
보코더를 위한 기능입니다** — 이 마커를 읽는 보코더는 ezv_for_diffsinger
뿐이며, 다른 보코더는 마커를 해석하지 못합니다.

dsdict.yaml 의 `symbols:` 타입을 기준으로 음소별 3분류 마커를 mel 최상위 빈에
찍습니다: **강제 무성**(AP/SP/exh 특례 + 무성 타입 자음), **강제 유성**(모음 +
항상 유성인 자음), **자율**(마커 인식 보코더가 프레임별로 스스로 판단 —
한국어 평음 ㄱ·ㄷ·ㅂ·ㅈ 처럼 유·무성이 오가는 음소용). dsdict 타입은 이렇게
정리하세요:

- 무성자음 → `stop` / `fricative` / `affricate` / `aspirate`
- 항상 유성인 자음(비음·유음 등) → `nasal` / `liquid` / `voiced`
- 위치에 따라 유/무성이 달라지는 자음(한국어 평음 등) → `lenis`

⚠️ **uv 마크된 어쿠스틱을 일반 보코더(nsf_hifigan /
pc_nsf_hifigan)로 렌더하면 안 됩니다** — mel 을 그대로 렌더하므로 마커
프레임마다 16kHz 삐 소리가 납니다. **패치 전에 원본 보이스뱅크를
백업**해두고, 다른 보코더로 돌아갈 땐 dsconfig 의 acoustic 을 원본으로
되돌리세요. 드래그&드롭 bat 에서 이 단계는 물어보고 기본값은 아니오입니다.

참고: 유/무성 테이블은 **패치 시점에 onnx 안에 구워집니다**. 이후 dsdict 타입을
수정했다면 (마커 없는 파일에) 패치를 다시 실행하세요 — 이미 패치된 onnx 는 사전
수정을 따라가지 않습니다.

## 원리

PC 보코더 체계에서 **음색은 어쿠스틱 모델이 받는 f0**, **실제 음정은 보코더가 받는 f0**가
결정합니다. 이 도구는 어쿠스틱 ONNX의 f0 입력 앞에 작은 전처리 그래프(약 20개 노드)를
`onnx.compose`로 병합합니다. 모델 가중치는 건드리지 않으며 추가 연산량은 무시할 수준입니다.
음역 밖 피치는 어쿠스틱에게만 범위 안으로 이동해 전달되므로(상한 초과는 아래로, 하한 미달은
위로), 음역 밖 노트도 건강한 음색을 유지한 채 PC 보코더가 원래 음정으로 렌더합니다.

## 주의

- OpenUtau의 SHFC(톤 시프트) 커브와 같은 노트에 **겹쳐 쓰면 이중 적용**됩니다 — 하나만 쓰세요.
- 어쿠스틱 모델만 패치합니다 (variance/pitch 모델은 건드리지 않음).
- 문제가 생기면 `dsconfig.yaml` 만 되돌리면 즉시 원상복구됩니다.
