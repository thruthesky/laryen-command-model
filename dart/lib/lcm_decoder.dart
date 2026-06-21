/// 모델 헤드 출력(argmax 인덱스) → VoiceIntent JSON. lcm/schema.py decode_intent 1:1 포팅.
///
/// 라리엔 Flutter 앱이 onnxruntime 추론 결과(헤드별 logits → argmax/threshold)를 라리엔
/// parseVoiceCommand 가 읽는 action JSON 으로 바꾼다. `labels`/`headSpecs` 는 ssot.json 에서
/// 생성한 라벨 공간(golden_tokenize.json 에도 포함). say 는 비워 둔다(라리엔이 기본 요약 채움).
library;

const String _none = '<none>';

/// [heads] : 헤드명 → 예측값(single/binary = int, multi = `List<int>` 0/1).
/// [labels]: 헤드명 → 라벨 목록(인덱스→문자열).
Map<String, dynamic> decodeIntent(
    Map<String, dynamic> heads, Map<String, List<String>> labels) {
  String lab(String head, int idx) => labels[head]![idx];
  int ix(String head) => heads[head] as int;
  List<int> multi(String head) => (heads[head] as List).cast<int>();

  final action = labels['action']![ix('action')];
  final out = <String, dynamic>{'action': action};

  switch (action) {
    case 'move':
      final loc = lab('location', ix('location'));
      if (loc != _none) {
        out['location'] = loc;
      } else {
        final d = lab('direction', ix('direction'));
        if (d != _none) out['direction'] = double.parse(d);
      }
    case 'hunt':
      final loc = lab('location', ix('location'));
      if (loc != _none) out['location'] = loc;
      final mons = <String>[];
      final mh = multi('monsters');
      for (var i = 0; i < mh.length; i++) {
        if (mh[i] == 1) mons.add(labels['monsters']![i]);
      }
      if (mons.isNotEmpty) out['monsters'] = mons;
      final hh = lab('hunt_hp', ix('hunt_hp'));
      if (hh != _none) out['huntHpPotionPct'] = int.parse(hh);
      if (ix('retreat_to_safe') == 1) out['retreatToSafeZone'] = true;
      final rh = lab('retreat_hp', ix('retreat_hp'));
      if (rh != _none) out['retreatHpPct'] = int.parse(rh);
    case 'potion':
      final p = lab('potion', ix('potion'));
      if (p != _none) out['potion'] = p;
    case 'equip':
      final s = lab('gear_set', ix('gear_set'));
      final g = lab('gear_single', ix('gear_single'));
      if (s != _none) {
        out['set'] = s;
      } else if (g != _none) {
        out['gear'] = g;
      }
    case 'unequip':
      final s = lab('slot', ix('slot'));
      if (s != _none) out['slot'] = s;
    case 'auto_combat':
      final m = lab('mode', ix('mode'));
      if (m != _none) out['mode'] = m;
    case 'open_menu':
      final t = lab('target', ix('target'));
      if (t != _none) out['target'] = t;
    case 'auto_potion':
      final pots = <String>[];
      final ah = multi('auto_potions');
      for (var i = 0; i < ah.length; i++) {
        if (ah[i] == 1) pots.add(labels['auto_potions']![i]);
      }
      if (pots.isNotEmpty) out['potions'] = pots;
      out['enable'] = ix('auto_potion_enable') == 1;
  }
  return out;
}
