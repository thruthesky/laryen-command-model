/// LCM 3계층 classify(2차 SML 판정) — lcm/infer.py classify 의 Dart 포팅.
///
/// 라리엔 Flutter 앱이 이것으로 발화를 분류한다. **onnxruntime 추론만 콜백([InferFn])으로
/// 주입** 하면 토큰화·threshold 판정·fallback·decode 는 검증된 이 Dart 코드가 처리한다
/// (onnxruntime Dart 패키지 도입은 사용자 승인 차단지점 — INTEGRATION.md). py infer.py 와
/// 동치: 의미 문자 없음 → fallback / action==unknown 또는 conf<threshold → fallback / 그 외 sml.
library;

import 'dart:math' as math;

import 'lcm_decoder.dart';
import 'lcm_tokenizer.dart';

/// 추론 콜백 — (input_ids[padLen], attentionMask[padLen]) → 헤드명 → logits.
/// 라리엔은 onnxruntime 세션으로 구현. 테스트는 고정 logits 를 주입.
typedef InferFn = Map<String, List<double>> Function(
    List<int> inputIds, List<int> attentionMask);

final RegExp _meaningful = RegExp(r'[가-힣a-zA-Z0-9]');

// 다중동작 분할 — "착용/물약 + 후속" 연결어미를 종결로 복원(infer.py split_compound 와 동일).
const _compoundConn = [
  ['착용하고', '착용'], ['장착하고', '장착'], ['입고', '입어'],
  ['먹고', '먹어'], ['마시고', '마셔'], ['쓰고', '써'], ['빨고', '빨아'],
];

List<String>? splitCompound(String text) {
  for (final c in _compoundConn) {
    final marker = '${c[0]} ';
    final i = text.indexOf(marker);
    if (i > 0) {
      final first = (text.substring(0, i) + c[1]).trim();
      final rest = text.substring(i + marker.length).trim();
      if (rest.isNotEmpty) return [first, rest];
    }
  }
  return null;
}

class LcmClassifier {
  LcmClassifier({
    required this.tokenizer,
    required this.labels,
    required this.headSpecs, // [{name, kind}] — kind: single|multi|binary
    required this.infer,
    this.threshold = 0.8,
    this.padLen = 32,
    this.padId = 0,
    this.multiThreshold = 0.5,
  });

  final LcmTokenizer tokenizer;
  final Map<String, List<String>> labels;
  final List<Map<String, String>> headSpecs;
  final InferFn infer;
  final double threshold;
  final int padLen;
  final int padId;
  final double multiThreshold;

  static int _argmax(List<double> v) {
    var bi = 0;
    for (var i = 1; i < v.length; i++) {
      if (v[i] > v[bi]) bi = i;
    }
    return bi;
  }

  static double _softmaxMax(List<double> logits) {
    final m = logits.reduce(math.max);
    var sum = 0.0;
    for (final x in logits) {
      sum += math.exp(x - m);
    }
    return 1.0 / sum; // exp(max-m)=1 → 최댓값 확률 = 1/Σexp(x-m)
  }

  /// 3계층 2차 판정. 다중동작이면 actions 배열로 결합. {layer:'sml'|'fallback', ...}.
  Map<String, dynamic> classify(String text) {
    final parts = splitCompound(text);
    if (parts != null) {
      final subs = parts.map(_classifyOne).toList();
      if (subs.every((s) => s['layer'] == 'sml')) {
        return {
          'layer': 'sml',
          'confidence': subs.map((s) => s['confidence'] as double).reduce(math.min),
          'command': {
            'actions': [for (final s in subs) s['intent']],
            'say': '',
          },
        };
      }
      return {'layer': 'fallback', 'confidence': 0.0};
    }
    final r = _classifyOne(text);
    if (r['layer'] == 'sml') {
      return {
        'layer': 'sml',
        'confidence': r['confidence'],
        'command': {'actions': [r['intent']], 'say': ''},
      };
    }
    return r;
  }

  Map<String, dynamic> _classifyOne(String text) {
    if (!_meaningful.hasMatch(text)) {
      return {'layer': 'fallback', 'confidence': 0.0};
    }
    // 고정 길이 패딩 입력.
    final raw = tokenizer.encodeWithSpecial(text);
    final ids = List<int>.filled(padLen, padId);
    final mask = List<int>.filled(padLen, 0);
    final n = math.min(raw.length, padLen);
    for (var i = 0; i < n; i++) {
      ids[i] = raw[i];
      mask[i] = 1;
    }
    final logits = infer(ids, mask);

    final actionLogits = logits['action']!;
    final conf = _softmaxMax(actionLogits);
    final action = labels['action']![_argmax(actionLogits)];
    if (action == 'unknown' || conf < threshold) {
      return {'layer': 'fallback', 'confidence': conf};
    }
    // 헤드별 예측 → decode.
    final heads = <String, dynamic>{};
    for (final spec in headSpecs) {
      final name = spec['name']!;
      final kind = spec['kind']!;
      final l = logits[name]!;
      if (kind == 'single') {
        heads[name] = _argmax(l);
      } else if (kind == 'binary') {
        // binary 헤드는 logits 길이 1 — sigmoid>0.5.
        heads[name] = (1.0 / (1.0 + math.exp(-l[0])) > 0.5) ? 1 : 0;
      } else {
        heads[name] = [for (final x in l) (1.0 / (1.0 + math.exp(-x)) > multiThreshold) ? 1 : 0];
      }
    }
    final intent = decodeIntent(heads, labels);
    return {'layer': 'sml', 'confidence': conf, 'intent': intent};
  }
}
