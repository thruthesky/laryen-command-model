// Dart decodeIntent ↔ 파이썬 schema.decode_intent parity — golden 으로 검증.
//
// 모델 헤드 출력(argmax 인덱스) → VoiceIntent JSON 변환이 파이썬과 1:1 같음을 보장한다.
import 'dart:convert';
import 'dart:io';

import 'package:lcm_tokenizer/lcm_decoder.dart';
import 'package:test/test.dart';

void main() {
  final root = Directory.current.path;
  final base = root.endsWith('/dart') ? '$root/..' : root;
  final goldenFile = File('$base/artifacts/golden_tokenize.json');
  final hasGolden = goldenFile.existsSync();

  test('Dart decodeIntent == 파이썬 golden (heads→intent 1:1)', () {
    final golden = jsonDecode(goldenFile.readAsStringSync()) as Map;
    final labels = (golden['labels'] as Map).map(
        (k, v) => MapEntry(k as String, (v as List).cast<String>()));
    for (final c in (golden['decode'] as List)) {
      final heads = (c['heads'] as Map).cast<String, dynamic>();
      final want = (c['intent'] as Map).cast<String, dynamic>();
      final got = decodeIntent(heads, labels);
      expect(got, equals(want), reason: 'heads=$heads');
    }
  }, skip: !hasGolden ? 'golden 없음(export_golden.py 먼저)' : false);
}
