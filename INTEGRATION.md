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

→ **A 채택·레퍼런스 완성.** [lcm/bpe_ref.py](lcm/bpe_ref.py)가 외부 BPE 라이브러리 없이
순수 로직(byte_to_unicode 표 + GPT-2 정규식 pre-tokenize + merges 병합 + vocab 조회)으로
인코딩한다. `tests/test_bpe.py`가 HF tokenizers와 **1:1 parity**(한/영/혼용/숫자/기호/공백)
를 보장한다. dart는 이 파일을 그대로 포팅하면 된다:
- `add_prefix_space=false`(HF ByteLevelBPETokenizer 기본과 일치 — 검증된 값).
- `bytes_to_unicode()` 표는 dart에서 동일하게 생성(0~255 고정 매핑).
- GPT-2 정규식 → dart `RegExp(unicode: true)`. `\p{L}`/`\p{N}` 유니코드 속성 지원.
- 검증: `scripts/export_golden.py`가 만든 `golden_tokenize.json`(text→ids)로 dart 단위테스트.

## 배포 — 모델 파일은 서버 다운로드

`lcm.int8.onnx`(~650KB) + `vocab.json`/`merges.txt`를 sherpa 모델처럼
`laryen.com/models/lcm/`에서 다운로드(앱 재배포 없이 갱신). fast-path 규칙
(`fast_path.json`)과 동일한 stale-while-revalidate 패턴.

## 통합 현황 & 남은 단계

LCM repo 측(모델·dart 토큰화/디코드)은 **완성**. 라리엔 lib 측만 남았고, 그 진입이
**사용자 승인 차단지점**이다.

- [x] **dart ByteLevelBPE** — `dart/lib/lcm_tokenizer.dart` + parity(`dart test`). (iter9)
- [x] **라벨 공간** — `golden_tokenize.json`의 `labels`/`head_specs`(ssot.json 파생). (iter5/10)
- [x] **decode_intent dart** — `dart/lib/lcm_decoder.dart` + parity. (iter10)
- [x] **3계층 classify dart** — `dart/lib/lcm_classifier.dart`(meaningful 가드·threshold·
  argmax→decode·fallback). 추론은 `InferFn` 콜백으로 추상화 → 라리엔은 그 함수만 채우면 됨.
  모킹 테스트 통과(`dart test`). (iter16)
- [ ] **InferFn 구현(onnxruntime 추론)** — 🛑 **사용자 승인 차단지점**(이것만 남음):
  - **(1) 비공식 패키지**: onnxruntime Dart 바인딩(pub.dev `onnxruntime` 등)은 Flame 공식이
    아니므로 CLAUDE.md상 도입 전 사용자 확인 필수. (sherpa_onnx가 ORT 세션을 재노출하면
    추가 패키지 0일 수 있으나 확인 필요.)
  - **(2) lib/ 변경 + flutter 빌드 + DTD 검증**: `voice_command_sheet.dart`의 fast-path 실패
    경로에 LCM 2차(`classify`)를 끼우고, 실패 시 기존 CF 폴백. 클라 빌드 영향 → DTD 시각 검증
    의무 + 다른 세션 working tree 충돌 주의.
  - **(3) 모델 서버 배포**: `lcm.int8.onnx`+`vocab.json`/`merges.txt`를 `laryen.com/models/lcm/`
    에(sherpa 모델·fast_path.json과 동일 stale-while-revalidate).

→ 위 3가지를 사용자가 승인하면, 라리엔은 [토큰화 lcm_tokenizer] + [추론 onnxruntime] +
  [디코드 lcm_decoder] + [3계층 classify(lcm_classifier)]를 조립해 통합 완료.

## InferFn 구현 예시 (사용자 onnxruntime 승인 후 — 복붙 착수용)

`dart/lib/lcm_classifier.dart`의 `InferFn`만 onnxruntime으로 채우면 끝이다:

```dart
// pubspec.yaml: onnxruntime: ^1.4  ← 🛑 비공식 패키지(사용자 승인 차단지점)
import 'package:onnxruntime/onnxruntime.dart';

// 1) 세션 1회 로드(서버 다운로드한 lcm.int8.onnx).
final session = OrtSession.fromFile(File(modelPath), OrtSessionOptions());
final headNames = headSpecs.map((s) => s['name']!).toList(); // golden head_specs 순서

// 2) InferFn = onnxruntime 추론.
InferFn ortInfer = (ids, mask) {
  final shape = [1, ids.length];
  final inputs = {
    'input_ids': OrtValueTensor.createTensorWithDataList(
        Int64List.fromList(ids), shape),
    'attention_mask': OrtValueTensor.createTensorWithDataList(
        Int64List.fromList(mask), shape),
  };
  final outs = session.run(OrtRunOptions(), inputs);
  final result = <String, List<double>>{};
  for (var i = 0; i < headNames.length; i++) {
    // 각 출력은 [1, L] (binary 는 [1]); 첫 배치만 double 리스트로.
    final v = (outs[i]!.value as List).first;
    result[headNames[i]] = v is List ? List<double>.from(v) : [v as double];
  }
  return result;
};

// 3) classify.
final clf = LcmClassifier(
    tokenizer: tk, labels: labels, headSpecs: headSpecs, infer: ortInfer);
final res = clf.classify(utterance); // {layer:'sml'|'fallback', command?}
```

라리엔 `voice_command_sheet.dart` 통합 지점(fast-path 실패 직후):
```dart
final r = clf.classify(text);
if (r['layer'] == 'sml') {
  // 2차 SML — 즉시 실행(좌표는 위치 id 로 SSOT 조회)
  game.executeVoiceCommand(r['command']); // parseVoiceCommand 형식
} else {
  // 3차 — 기존 CF 폴백
  await classifyVoiceAssistantRemote(text: text, ...);
}
```
