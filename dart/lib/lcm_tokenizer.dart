/// 라리엔 Command Model — ByteLevelBPE 토크나이저(순수 Dart).
///
/// lcm/bpe_ref.py 의 1:1 포팅. 라리엔 Flutter 앱이 sherpa_onnx 와 같은 onnxruntime 으로
/// LCM 을 추론하기 전, 발화를 *모델과 동일한* 토큰 id 로 바꾼다. HF tokenizers(Rust)는
/// Flutter 에 없으므로 이 순수 Dart 구현을 쓴다. tests/ 의 golden(파이썬 생성)과 1:1 일치.
library;

import 'dart:convert';
import 'dart:io';

/// GPT-2 ByteLevel pre-tokenize 정규식(HF ByteLevel 기본과 동일).
final RegExp _pat = RegExp(
  r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
  unicode: true,
);

/// 0~255 byte → 가시 유니코드 char(GPT-2 표준). 파이썬 bytes_to_unicode 와 동일.
Map<int, String> bytesToUnicode() {
  final bs = <int>[];
  for (var i = '!'.codeUnitAt(0); i <= '~'.codeUnitAt(0); i++) {
    bs.add(i);
  }
  for (var i = '¡'.codeUnitAt(0); i <= '¬'.codeUnitAt(0); i++) {
    bs.add(i);
  }
  for (var i = '®'.codeUnitAt(0); i <= 'ÿ'.codeUnitAt(0); i++) {
    bs.add(i);
  }
  final cs = List<int>.from(bs);
  var n = 0;
  for (var b = 0; b < 256; b++) {
    if (!bs.contains(b)) {
      bs.add(b);
      cs.add(256 + n);
      n++;
    }
  }
  final m = <int, String>{};
  for (var i = 0; i < bs.length; i++) {
    m[bs[i]] = String.fromCharCode(cs[i]);
  }
  return m;
}

class LcmTokenizer {
  LcmTokenizer(this.encoder, this.bpeRanks, {this.addPrefixSpace = false})
      : byteEncoder = bytesToUnicode();

  final Map<String, int> encoder; // token → id
  final Map<String, int> bpeRanks; // "a b" → rank
  final Map<int, String> byteEncoder;
  final bool addPrefixSpace;

  static LcmTokenizer load(String vocabPath, String mergesPath) {
    final vocab = (jsonDecode(File(vocabPath).readAsStringSync()) as Map)
        .map((k, v) => MapEntry(k as String, v as int));
    final lines = File(mergesPath).readAsLinesSync();
    final ranks = <String, int>{};
    var rank = 0;
    for (final ln in lines) {
      if (ln.isEmpty || ln.startsWith('#')) continue;
      ranks[ln] = rank++;
    }
    return LcmTokenizer(vocab, ranks);
  }

  List<String> _bpe(String token) {
    var word = token.split('');
    if (word.length < 2) return word;
    while (true) {
      // 최소 rank 인접 쌍 찾기.
      int bestRank = 1 << 30;
      int bestI = -1;
      for (var i = 0; i < word.length - 1; i++) {
        final r = bpeRanks['${word[i]} ${word[i + 1]}'];
        if (r != null && r < bestRank) {
          bestRank = r;
          bestI = i;
        }
      }
      if (bestI < 0) break;
      final first = word[bestI], second = word[bestI + 1];
      final next = <String>[];
      var i = 0;
      while (i < word.length) {
        if (i < word.length - 1 && word[i] == first && word[i + 1] == second) {
          next.add(first + second);
          i += 2;
        } else {
          next.add(word[i]);
          i++;
        }
      }
      word = next;
      if (word.length == 1) break;
    }
    return word;
  }

  /// 텍스트 → 토큰 id(특수토큰 미포함).
  List<int> encode(String text) {
    var t = text;
    if (addPrefixSpace && t.isNotEmpty && !RegExp(r'^\s').hasMatch(t)) t = ' $t';
    final ids = <int>[];
    final unk = encoder['<unk>'] ?? 3;
    for (final m in _pat.allMatches(t)) {
      final piece = m.group(0)!;
      final tok = utf8.encode(piece).map((b) => byteEncoder[b]!).join();
      for (final sub in _bpe(tok)) {
        ids.add(encoder[sub] ?? unk);
      }
    }
    return ids;
  }

  /// `<s> ... </s>` 래핑(모델 입력 형식 — HF post_processor 와 동일).
  List<int> encodeWithSpecial(String text) {
    final bos = encoder['<s>'] ?? 1;
    final eos = encoder['</s>'] ?? 2;
    return [bos, ...encode(text), eos];
  }
}
