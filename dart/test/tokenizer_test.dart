// Dart ByteLevelBPE ↔ 파이썬(HF tokenizers) parity — golden 으로 검증.
//
// 라리엔 Flutter 앱이 이 Dart 토크나이저로 LCM 추론 입력을 만들 수 있음을 증명한다.
// golden_tokenize.json 은 scripts/export_golden.py 가 생성(bpe_ref==HF tokenizers).
import 'dart:convert';
import 'dart:io';

import 'package:lcm_tokenizer/lcm_tokenizer.dart';
import 'package:test/test.dart';

void main() {
  // dart test 의 cwd = 패키지 루트(dart/). 산출물은 상위(..)에 있다.
  final root = Directory.current.path;
  final base = root.endsWith('/dart') ? '$root/..' : root;
  final tkDir = '$base/artifacts/tokenizer';
  final goldenFile = File('$base/artifacts/golden_tokenize.json');

  test('golden 산출물 존재(없으면 export_golden.py 먼저)', () {
    expect(File('$tkDir/vocab.json').existsSync(), isTrue,
        reason: 'python -m lcm.train + scripts/export_golden.py 먼저 실행');
    expect(goldenFile.existsSync(), isTrue);
  }, skip: !File('$tkDir/vocab.json').existsSync()
      ? '토크나이저/golden 없음' : false);

  test('Dart BPE == 파이썬 golden (text→ids 1:1)', () {
    final tk = LcmTokenizer.load('$tkDir/vocab.json', '$tkDir/merges.txt');
    final golden = jsonDecode(goldenFile.readAsStringSync()) as Map;
    final cases = golden['tokenize'] as List;
    for (final c in cases) {
      final text = c['text'] as String;
      final want = (c['ids'] as List).cast<int>();
      final got = tk.encodeWithSpecial(text);
      expect(got, equals(want), reason: "'$text'");
    }
  }, skip: !File('$tkDir/vocab.json').existsSync()
      ? '토크나이저/golden 없음' : false);
}
