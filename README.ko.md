# DiffSinger 어쿠스틱 한계음 패치

[English](README.md) | **한국어**



https://github.com/user-attachments/assets/0ac279cf-2716-45b8-bb12-f3ded3a8219c



DiffSinger 보이스뱅크의 **학습 음역을 넘는 고음**에서 음색이 무너지는 문제를,
어쿠스틱 ONNX 파일에 "한계음" 전처리를 이식해서 해결하는 도구입니다.

- 한계음을 넘는 구간은 **어쿠스틱에게만 낮춰서** 들려줌 — 음색은 항상 학습 범위 안에서 생성
- 실제 음정은 **PC(pitch controllable) 보코더**가 원래 f0 그대로 렌더
- OpenUtau에서 SHFC(톤 시프트) 커브를 손으로 그리는 것과 같은 효과를 **자동으로**, 옥타브 제한 없이

> ⚠️ **반드시 PC 보코더와 함께 사용하세요** (예: `pc_nsf_hifigan`).
> 일반 보코더로 렌더하면 고음의 음정 자체가 내려가 버립니다.

## 설치

Python 3.8 이상 + 아래 한 줄:

```
pip install onnx numpy
```

(수치 self-test까지 보려면 `pip install onnxruntime` 추가 — 없어도 패치는 됩니다.)

## 사용법

```
python patch_acoustic_limit.py <보이스뱅크 폴더>/acoustic.onnx --limit D5
```

- `acoustic.limitD5.onnx` 가 생성됩니다.
- 보이스뱅크의 `dsconfig.yaml` 을 열어 `acoustic:` 항목을 새 파일명으로 바꾸면 끝.
- 되돌리기: `acoustic:` 항목을 원래 파일명으로 복원 (원본 파일은 수정되지 않습니다).
- Windows에서는 `drag_drop_patch.bat` 에 acoustic.onnx 를 **끌어다 놓아도** 됩니다.

한계음 표기: `D5`, `F#4`, `Bb3`, 또는 MIDI 번호(`74`). C4 = 60 기준.

### 옵션

| 옵션 | 설명 |
|---|---|
| `--mode transpose` (기본) | 한계음 초과 구간을 **반음 단위로 통째로 내림** — 비브라토·피치 굴곡 보존. 옥타브 제한 없음 |
| `--mode clamp` | 단순 `min(f0, 한계음)` — 한계 위 비브라토가 평탄해짐 |
| `--inplace` | 원본 파일을 직접 교체 (원본은 `.bak` 으로 백업) |
| `--f0_input <이름>` | f0 입력 이름이 `f0` 가 아닌 모델용 |
| `--win N` | transpose 판단용 이동평균 길이 (기본 21프레임 ≈ 250ms) |

## 원리

PC 보코더 체계에서 **음색은 어쿠스틱 모델이 받는 f0**, **실제 음정은 보코더가 받는 f0**가
결정합니다. 이 도구는 어쿠스틱 ONNX의 f0 입력 앞에 작은 전처리 그래프(약 20개 노드)를
`onnx.compose`로 병합합니다. 모델 가중치는 건드리지 않으며 추가 연산량은 무시할 수준입니다.
한계음을 넘는 피치는 어쿠스틱에게만 낮춰 전달되므로, 음역 밖 노트도 건강한 음색을 유지한 채
PC 보코더가 원래 음정으로 렌더합니다.

## 주의

- OpenUtau의 SHFC(톤 시프트) 커브와 같은 노트에 **겹쳐 쓰면 이중 적용**됩니다 — 하나만 쓰세요.
- 어쿠스틱 모델만 패치합니다 (variance/pitch 모델은 건드리지 않음).
- 문제가 생기면 `dsconfig.yaml` 만 되돌리면 즉시 원상복구됩니다.
