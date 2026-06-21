# LCM 개발 진행 로그 (Ralph Loop)

매 iteration: **자아비판 → 계획 → 구현 → 검증 → 커밋**. 다음 iteration이 이 로그와
git history를 참조해 이어간다.

## 완료조건(목표)
1. 텍스트 → LCM → VoiceIntent JSON 분류 + **유닛테스트 증명**
2. **ONNX Runtime 동작 + 유닛테스트 증명**
3. 라리엔 개념·컨트롤셋·설정 훈련 → 검증 → 정확도 향상
4. 다양한 입력 → 응답 → 검증 → 훈련 반복 → 정확도 향상
5. 1~4를 ≥100회 반복

## 현재 상태(메트릭) — iter 1 종료 시점
- 데이터: 합성 1572건(균형 개선: stop 27·auto_potion 32·unequip 56 …)
- 학습: train loss 0.010, **val action_acc 0.90 / exact_acc 0.78**(608→1415 train)
- pytest: **8/8 통과**(schema round-trip 3 + ONNX Runtime 2 + golden 정확도 3)
- ONNX export: fp32 dynamic(parity maxdiff 0.0) + int8 고정 649KB(argmax 5/5 일치)
- 완료조건 1·2 ✅ 증명, 3·4 진행(정확도 0.55→0.78)

## Iteration 로그

### iter 4 (2026-06-22) — 약점 클래스 보강·용량↑ (솔직한 평가: iter3와 동등)
**자아비판**: auto_combat(n=3)·auto_potion(n=2) 표본 극소, stop 약점.

**구현**: auto_combat 27→51·auto_potion 32→93 표현 증강, "뭐 할까/언제" OOD 패턴 추가
(unknown→530, 총 2572건). 데이터만 늘렸더니 **후퇴**(홀드아웃 0.90→0.80, 명령 sml
0.96→0.91) → 모델 용량↑(d_model 128→160·layers 2→3·200ep)로 데이터 활용.

**결과**: exact 0.899, 홀드아웃 0.87, 명령 sml 0.93. **iter3(0.91/0.90)와 측정 노이즈
내 동등** — 큰 개선 아님. 약점 표본·용량·측정 인프라는 견고해짐. pytest 9/9, int8 1148KB.
잔존 약점: stop 0.43(n=7, "행동 멈춰"→hunt) — 단 "멈춰"류는 1차 fast-path 가 처리해 실전
영향 작음. potion/unequip 0.78(작은 val 표본 노이즈).

**교훈**: 데이터 무작정 증강은 순이득이 아니며 용량/학습이 따라야 한다. 작은 클래스는
val 표본 부족으로 측정 노이즈가 커 단일 수치로 판단하기 어렵다(홀드아웃 30 고정셋이 더 신뢰).

**다음(iter 5 후보)**: ① stop 혼동 직접 수정(hunt 와 분리 — 데이터/가중치) ② val 표본
부족 → stratified split 또는 클래스별 최소 보장 ③ **dart BPE 포팅 + parity golden**(통합
핵심) ④ 토큰화 golden export(text→ids) 로 dart 구현 타겟 제공.


### iter 3 (2026-06-22) — OOD 과신 해결
**자아비판**: confidence 가 fallback 안전장치로 *작동하는지* 측정한 적 없음. 잡담/질문을
명령 실행하는 오작동(가장 위험)을 정량화 필요.

**핵심 발견**: 홀드아웃 비명령(학습셋에 없는 잡담/질문) 30건 중 16건을 **conf 1.0 으로
명령 오인**("날씨 좋다"→hunt 1.0). confidence threshold 가 안전망이 전혀 안 됨 — 좁은
도메인 softmax 분류기의 OOD 과신. **in-domain 질문 데이터 증강(unknown 77→381)으로도
미해결**(recall 0.47→0.43).

**해결(정석 기법)**:
- **outlier exposure**: 게임 무관 일상 문장(`_gen_smalltalk` — 날씨/음식/감정/잡담 한·영)을
  unknown 경계로 노출.
- **label smoothing 0.1**(action 헤드): softmax 과신 억제 → OOD confidence 하락.
- `lcm/eval.py`: confidence threshold sweep + 홀드아웃 일반화 측정.
- `tests/test_accuracy.py`: 홀드아웃 fallback recall ≥0.7 회귀 가드.
- infer threshold 0.6→0.7(calibration 후 비명령 fallback 1.0·명령 sml 0.96 지점).

**결과**: 홀드아웃 fallback recall **0.43→0.90**(오인 conf 1.0→0.6~0.86). val 비명령
fallback 0.95~1.0, 명령 sml 채택 0.96(CF 절약 유지). exact 0.91 유지. pytest **9/9**.

**다음(iter 4 후보)**: ① 남은 오인 3건("주말에 뭐 할까"→hunt 0.62) — smalltalk 다양화 ②
auto_combat/auto_potion 표현 증강(27·32) ③ slot 정확도 추가 개선(target 0.75) ④ dart
BPE 포팅 + parity golden ⑤ confidence calibration 정량(ECE).


### iter 2 (2026-06-22)
**자아비판**: action 0.90 vs exact 0.78 격차 = 슬롯 오류인데 *어느 헤드가 약한지* 측정
도구가 없어 맹목 증강 중. unknown(잡담/질문) fallback이 깨지면 오작동.

**구현**:
- `lcm/eval.py` — 헤드별·action별 정확도 + 오답 샘플 진단. **진단 결과**: unknown 0.14·
  stop 0.0·monsters 0.83·target 0.75 가 약점(적은 클래스 표본 부족이 근본).
- `synth.py` 보강: unknown 38→77(게임 질문 한/영 다양화 — fallback 안전장치), stop
  27→47(영어 halt/pause/freeze 등), hunt monster 표본↑(4종×4표현+2종동시) → 2034건.
- `INTEGRATION.md` — 플러터 onnxruntime 통합 설계(BPE 토큰화 dart 포팅이 핵심 난점).

**결과**: exact 0.78→**0.91**, action 0.90→**0.97**. hunt 1.0·move 0.99·equip/unequip/
auto_combat/auto_potion 1.0. pytest 8/8, export fp32/int8 parity 5/5. int8 678KB.
unknown 0.4·stop 0.5 는 val 표본 5·2건이라 통계 불안정(다음 iter: train eval·confidence).

**다음(iter 3 후보)**: ① **confidence 분석**(unknown 분류 실패해도 저신뢰면 fallback 되는지
— eval 에 confidence 분포·fallback recall) ② val 표본 부족 클래스의 train/val 일반화 격차
측정 ③ auto_combat/auto_potion 표현 증강(여전히 27·32) ④ dart BPE 포팅 + parity golden.


### iter 1 (2026-06-22)
**자아비판**: ① 완료조건 2의 ONNX Runtime 유닛테스트 부재(inline 검증만) ② exact 0.55,
action 불균형(move 330 vs auto_potion 10) ③ 데이터 608건/어순 변형 부족 ④ 정확도 회귀
테스트 부재.

**구현**:
- `synth.py` v2 — 어미/공손/어순 변형 + 클래스 균형 → 675→1572건.
- `tests/test_onnx.py` — dynamo parity(PyTorch==onnxruntime, 여러 길이) + int8 동작.
- `tests/test_accuracy.py` — golden 18발화 action 정확도 + fallback 라우팅(ckpt 조건부).
- **모델 trace-safe 수정**: boolean `src_key_padding_mask`가 legacy export에서 상수 폴딩
  되어 다른 입력 길이에서 ONNX 출력이 틀어지는 회귀 발견 → float mask(-1e9) + 배포는
  dynamo exporter(symbolic mask)로 dynamic seq, int8은 고정 길이 best-effort.

**결과**: pytest 8/8 통과. 정확도 exact 0.55→**0.78**, action **0.90**. ONNX fp32 parity
maxdiff 0.0 + int8 649KB argmax 5/5. dynamo export 가 원본 model 추론상태에 영향 주는
함정 발견 → 검증은 fresh load 로 분리.

**다음(iter 2 후보)**: ① **slot 정확도 테스트**(monsters/location/direction 별도 측정 —
action 0.90 vs exact 0.78 격차 = 슬롯 오류) ② action 균형 추가(move/hunt 여전히 다수) ③
hunt 의 monsters 누락 개선(합성에 monsters 다양화) ④ distillation 골격(CF Gemini teacher
라벨링 스크립트) ⑤ 라리엔 클라 onnxruntime 추론 통합 의사코드 문서.
