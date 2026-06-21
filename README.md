# 라리엔 명령어 모델 (Laryen Command Model, LCM)

라리엔 게임의 **음성/텍스트 명령**을 이해해 *온디바이스*에서 즉시 처리하는 초경량 SML.
발화 텍스트를 라리엔 `VoiceIntent` JSON(`{"actions":[...]}`)으로 **분류**한다(자유 텍스트
생성이 아님). 라리엔 클라가 `sherpa_onnx`와 같은 **onnxruntime**으로 추론하므로 추가
런타임이 필요 없다.

## 무엇을 하는가 — 3계층 디스패치

```
발화 텍스트(sherpa STT 결과 또는 키보드)
  │
  ├─ 1차 fast-path  : 모호성 0 단순 명령("멈춰"·"물약"·"메뉴"). 라리엔 클라의
  │                   FastPathRules(이미 구현·서버 다운로드)가 0ms 처리.
  │
  ├─ 2차 SML(본 모델): action + 슬롯을 분류해 VoiceIntent JSON 생성.
  │                   confidence(action softmax 최댓값) ≥ 임계값이고 action≠unknown → 채택.
  │                   → 라리엔이 위치 id로 좌표 SSOT 조회 후 게임 제어 / 서버 DSL 전송.
  │
  └─ 3차 CF Gemini  : SML이 unknown이거나 confidence < 임계값(복합·모호·처음 보는 발화)
                      → 기존 classifyVoiceAssistantRemote(text:...)로 *텍스트*를 CF에 폴백.
```

사용자 요청 대응:
- **1차 fast-path 수행** → `lcm/infer.py`의 `classify()`가 SML 판정 전 단계로 통합(실 fast-path는 라리엔 클라 `voice_fast_path.dart`가 이미 담당).
- **서버로 DSL 전송** → 2차 SML이 만든 `VoiceIntent`를 라리엔 `executeVoiceIntent`가 이동/사냥 INPUT·DSL로 서버에 전송.
- **1·2차 실패 시 CF Gemini 3.5 Flash** → 3차 폴백(`confidence < threshold` 또는 `unknown`).

## 설계 핵심

- **분류 ≠ 생성**: 출력이 고정 스키마(action enum 10종 + 슬롯)라 디코더로 JSON을 생성하지
  않고 멀티헤드 분류기로 각 슬롯을 예측한다 → 모델이 수 MB로 작고 환각이 없다.
- **`say` 필드는 모델 밖**: 사용자 음성 피드백(`say`)은 모델이 생성하지 않는다. intent가
  정해지면 라리엔 `_parseOneIntent`가 기본 요약을 채운다(환각 0).
- **1차 single-action**: 한 발화당 1개 action을 예측. 복합 명령("강철 입고 사냥")은
  confidence가 낮아 3차 CF로 폴백한다(점진적으로 multi-action 확장 가능).
- **SSOT는 라리엔 dart**: 위치/몬스터/장비/물약/메뉴/fast-path 별칭은 라리엔 `lib/`가
  진실이다. `scripts/sync_ssot.py`가 dart를 파싱해 `config/ssot.json`으로 뽑는다 —
  손-미러로 인한 drift 차단(신규 사냥터·몬스터 추가 시 재실행).
- **언어**: 한국어/영어 우선(코드스위칭 "plate 세트 입고" 포함). 바이트 레벨 BPE.
- **단일 머신**: M5 Apple Silicon(MPS) 한 대로 학습. 분산학습 없음.

## 디렉토리

```
laryen-command-model/
├── config/ssot.json          # 라리엔 dart 추출물(라벨 공간 SSOT) — sync_ssot.py가 생성
├── scripts/
│   ├── sync_ssot.py          # ../lib dart → config/ssot.json
│   └── gen_data.py           # 합성 데이터 → data/generated/{train,val}.jsonl
├── lcm/
│   ├── schema.py             # LabelSpace + intent ↔ 라벨 인코딩/디코딩
│   ├── synth.py              # SSOT + 한/영 템플릿 → (발화, intent) 페어
│   ├── tokenizer.py          # 바이트레벨 BPE (한/영)
│   ├── dataset.py            # jsonl → 텐서
│   ├── model.py              # LcmEncoder(인코더 + 멀티헤드 분류기) + loss
│   ├── train.py              # MPS 학습 루프
│   ├── infer.py              # 추론 + 3계층 classify()
│   └── export_onnx.py        # ONNX + int8 양자화 + onnxruntime 검증
├── tests/                    # schema round-trip 등
└── pyproject.toml            # uv
```

## 사용법

```bash
uv sync --extra dev                       # 의존성(torch/tokenizers/onnx…)

python scripts/sync_ssot.py               # 1) 라리엔 dart → config/ssot.json
python scripts/gen_data.py                # 2) 합성 데이터 생성
python -m lcm.train --epochs 25           # 3) M5(MPS) 학습 → checkpoints/lcm.pt
python -m lcm.infer "강남에서 캐스터 사냥"   # 4) 추론(3계층 classify 결과 JSON)
python -m lcm.export_onnx                  # 5) ONNX int8 → artifacts/lcm.int8.onnx
```

테스트(순수 파이썬 — torch 불필요):
```bash
python -m pytest tests/ -q
```

## 데이터 전략 (현재 → 다음)

1. **현재 — 합성**: SSOT 값 + 한/영 템플릿으로 (발화 → intent) 페어를 만든다(`synth.py`).
2. **다음 — distillation**: 운영 중인 CF Gemini를 *teacher*로, 3차 폴백한 실사용 발화를
   라벨링해 학습셋에 합류시킨다(L3 로그 → 선순환). 합성 데이터의 분포 편향을 보정한다.

## 라리엔 통합 (모델 완성 후)

- `artifacts/lcm.int8.onnx` + `artifacts/tokenizer/`를 sherpa 모델처럼 **서버 다운로드**
  (`laryen.com/models/...`)로 배포 → 앱 재배포 없이 모델 갱신.
- 클라는 `sherpa_onnx`의 onnxruntime으로 추론, `infer.py::classify()`와 동치 로직으로
  `sml`이면 즉시 `executeVoiceIntent`, `fallback`이면 `classifyVoiceAssistantRemote(text:)`.

## 출처

[`thruthesky/fai`](https://github.com/thruthesky/fai)의 BPE 토크나이저·PyTorch(MPS) 학습
경험을 재활용했다. 단 fai는 생성형 GPT + 분산학습이고, LCM은 **분류기 + 단일 머신**이라
fai의 `distributed/`(BOINC/federated)는 채택하지 않았다.
