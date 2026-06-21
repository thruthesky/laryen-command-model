# 라리엔 클라(Flutter) onnxruntime 통합 설계

LCM(`lcm.int8.onnx`)을 라리엔 Flutter 앱에서 온디바이스 추론하는 방법. 완료조건 "플러터에서
ONNX Runtime 내장 가능"의 설계 SSOT. **3계층 디스패치**(fast-path → SML → CF Gemini)의
2차를 클라에서 수행한다.

## 런타임 — onnxruntime은 이미 있다

라리엔은 sherpa_onnx STT가 **onnxruntime**을 이미 들고 있다(iOS/Android/macOS/Windows
네이티브). LCM도 같은 엔진으로 돌리므로 **추가 네이티브 의존성 0**.
- pub.dev `onnxruntime` 패키지(Dart FFI 바인딩) 또는 sherpa가 노출하는 ORT 세션 재사용.
- 입력: `input_ids`(int64, [1, L]), `attention_mask`(int64, [1, L]). 고정 L=32 권장(PAD_LEN).
- 출력: 15개 헤드 logits. `action` argmax + confidence(softmax max)로 3계층 판정.

## 추론 파이프라인 (의사코드)

```dart
// 1) 발화 텍스트 → 토큰 id (아래 §BPE)
final ids = lcmTokenizer.encode(text);            // <s> ... </s>, 최대 32, pad=0
final attn = List.filled(32, 0)..setRange(0, ids.length, 1);
final input = pad(ids, 32, padId: 0);

// 2) onnxruntime 추론
final outputs = ortSession.run({'input_ids': int64[1,32], 'attention_mask': int64[1,32]});

// 3) action argmax + confidence
final actionLogits = outputs['action'];           // [1, 10]
final conf = softmax(actionLogits).reduce(max);
final action = kActions[argmax(actionLogits)];

// 4) 3계층 판정 (lcm/infer.py::classify 와 동치)
if (action == 'unknown' || conf < 0.6) {
  // 3차: 기존 classifyVoiceAssistantRemote(text: text) 로 CF Gemini 폴백
} else {
  // 2차 채택: 각 슬롯 헤드 argmax → VoiceIntent 조립 → executeVoiceIntent
  final intent = decodeHeads(outputs);             // schema.decode_intent 와 동치
}
```

슬롯 디코딩 규칙은 [lcm/schema.py](lcm/schema.py)의 `decode_intent`와 **반드시 1:1**로
맞춘다(action별 활성 슬롯, `<none>` 처리, direction 버킷→각도, monsters 멀티라벨).
라벨 목록은 `config/ssot.json`을 빌드시 dart 상수로 변환해 공유한다(SSOT 단일화).

## 🔑 핵심 난점 — BPE 토큰화를 dart에서

LCM은 **ByteLevelBPE**(`artifacts/tokenizer/{vocab.json,merges.txt}`)를 쓴다. sherpa STT의
토크나이저와 별개라 **dart에서 같은 BPE를 재현**해야 추론이 맞다. 옵션:

| 방안 | 설명 | 평가 |
|---|---|---|
| **A. dart BPE 포팅** | vocab.json+merges.txt를 dart ByteLevelBPE 구현으로 인코딩 | 권장 — 가볍고 정확. byte-level이라 한/영 무손실 |
| B. 토크나이저도 ONNX화 | HF tokenizers→ONNX(onnxruntime-extensions) | extensions 네이티브 추가 필요(무게↑) |
| C. char/byte 직접 입력 | BPE 없이 byte 임베딩 모델로 재설계 | 모델 변경, vocab↑, 다음 검토 |

→ **A 채택 예정.** ByteLevelBPE는 ① 텍스트를 UTF-8 byte로 → ② byte→unicode 매핑 →
③ merges 순서대로 병합. dart 구현 + `tests/`의 golden(텍스트→id)으로 파이썬과 일치 검증.

## 배포 — 모델 파일은 서버 다운로드

`lcm.int8.onnx`(~650KB) + `vocab.json`/`merges.txt`를 sherpa 모델처럼
`laryen.com/models/lcm/`에서 다운로드(앱 재배포 없이 갱신). fast-path 규칙
(`fast_path.json`)과 동일한 stale-while-revalidate 패턴.

## 다음 작업(통합 단계)
1. dart ByteLevelBPE 인코더 + 파이썬 parity golden 테스트.
2. `config/ssot.json` → dart 라벨 상수 생성 스크립트.
3. `decode_intent` dart 포팅 + 단위 테스트(파이썬 케이스 재사용).
4. onnxruntime 세션 로드 + 3계층 classify 위젯 통합.
