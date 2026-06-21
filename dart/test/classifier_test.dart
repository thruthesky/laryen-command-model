// LcmClassifier 3계층 판정 로직 검증(추론은 모킹 — logits→판정/decode 로직만).
// onnxruntime parity 는 파이썬 test_onnx 가 보장하므로, 여기선 classify 의 흐름
// (meaningful 가드·threshold·argmax→decode·fallback)을 검증한다.
import 'dart:convert';
import 'dart:io';

import 'package:lcm_tokenizer/lcm_classifier.dart';
import 'package:lcm_tokenizer/lcm_tokenizer.dart';
import 'package:test/test.dart';

void main() {
  final root = Directory.current.path;
  final base = root.endsWith('/dart') ? '$root/..' : root;
  final tkDir = '$base/artifacts/tokenizer';
  final goldenFile = File('$base/artifacts/golden_tokenize.json');
  final has = File('$tkDir/vocab.json').existsSync() && goldenFile.existsSync();

  late LcmTokenizer tk;
  late Map<String, List<String>> labels;
  late List<Map<String, String>> headSpecs;

  if (has) {
    tk = LcmTokenizer.load('$tkDir/vocab.json', '$tkDir/merges.txt');
    final g = jsonDecode(goldenFile.readAsStringSync()) as Map;
    labels = (g['labels'] as Map)
        .map((k, v) => MapEntry(k as String, (v as List).cast<String>()));
    headSpecs = (g['head_specs'] as List)
        .map((e) => (e as Map).map((k, v) => MapEntry(k as String, v as String)))
        .toList();
  }

  // 특정 action 에 큰 logit 을 주는 모킹 추론(나머지 헤드는 <none>/0).
  InferFn mock(String action, {double actionLogit = 12.0}) {
    return (ids, mask) {
      final out = <String, List<double>>{};
      for (final spec in headSpecs) {
        final name = spec['name']!;
        if (spec['kind'] == 'binary') {
          out[name] = [-12.0]; // sigmoid→0
        } else {
          out[name] = List<double>.filled(labels[name]!.length, 0.0);
        }
      }
      out['action']![labels['action']!.indexOf(action)] = actionLogit;
      return out;
    };
  }

  LcmClassifier make(InferFn infer) => LcmClassifier(
      tokenizer: tk, labels: labels, headSpecs: headSpecs, infer: infer);

  test('의미 문자 없으면 fallback(추론 호출 안 함)', () {
    var called = false;
    final c = make((ids, mask) {
      called = true;
      return {};
    });
    for (final t in ['', '  ', '!!!', '...', '😀']) {
      expect(c.classify(t)['layer'], 'fallback', reason: "'$t'");
    }
    expect(called, isFalse);
  }, skip: !has ? 'golden/토크나이저 없음' : false);

  test('높은 conf 명령 → sml + decode', () {
    final c = make(mock('stop'));
    final r = c.classify('멈춰');
    expect(r['layer'], 'sml');
    expect(r['command']['actions'][0]['action'], 'stop');
  }, skip: !has ? 'golden/토크나이저 없음' : false);

  test('action=unknown → fallback', () {
    final c = make(mock('unknown'));
    expect(c.classify('이건 뭐지')['layer'], 'fallback');
  }, skip: !has ? 'golden/토크나이저 없음' : false);

  test('낮은 conf(균등 logit) → fallback', () {
    final c = make((ids, mask) {
      final out = <String, List<double>>{};
      for (final spec in headSpecs) {
        out[spec['name']!] = spec['kind'] == 'binary'
            ? [-12.0]
            : List<double>.filled(labels[spec['name']!]!.length, 0.0); // 전부 0 → 균등
      }
      return out;
    });
    // action logits 전부 0 → softmax max = 1/len < 0.8 → fallback.
    expect(c.classify('강남 사냥')['layer'], 'fallback');
  }, skip: !has ? 'golden/토크나이저 없음' : false);
}
