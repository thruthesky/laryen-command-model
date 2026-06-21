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
