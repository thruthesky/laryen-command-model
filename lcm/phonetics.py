"""STT 음소 혼동 시뮬레이션 — 한글 자모(초/중/종성)를 *음성학적으로 가까운* 자모로 교체해
실제 STT(SenseVoice) 전사 오류를 모사한다. synth(데이터 생성)·dataset(train augment) 공유.

기존 augment 의 한계(받침 탈락 + 모음 ±1 인덱스)를 보완 — 실제 폴백을 일으킨 오류(LCM v2
plan §1 측정: "강남→간남"(받침 ㅇ→ㄴ)·"멈춰→먼춰"(초성 ㅁ→ㄴ)·"문약"(받침 ㄹ→ㄴ))는
*탈락이 아니라 혼동 교체*다. 음운 그룹 안에서 서로 바꿔 같은 라벨로 학습 → 음소 robust.
"""
from __future__ import annotations

import random as _random

# 초성(19) 인덱스: ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ
# 조음위치/조음방법이 가까워 STT 가 헷갈리는 그룹(연구개·치경·양순·경구개·치찰·유음↔비음).
_CHO_GROUPS = [[0, 1, 15], [3, 4, 16], [7, 8, 17], [12, 13, 14], [9, 10], [2, 5]]

# 중성(21): ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ
# 청취 혼동이 잦은 모음 페어(ㅓ↔ㅗ·ㅐ↔ㅔ·ㅡ↔ㅜ·ㅕ↔ㅔ·ㅏ↔ㅑ·ㅗ↔ㅘ).
_JUNG_GROUPS = [[4, 8], [1, 5], [18, 13], [6, 5], [0, 2], [8, 9]]

# 종성(28, 0=받침없음): 비음(ㄴㅇㅁ)·폐쇄(ㄱㄷㅂㅅ)·유음↔비음(ㄹㄴ) 혼동.
_JONG_GROUPS = [[4, 21, 16], [1, 7, 17, 19], [8, 4], [22, 19, 7]]


def _pick(idx: int, groups: list[list[int]], rng) -> int:
    for g in groups:
        if idx in g:
            alts = [x for x in g if x != idx]
            if alts:
                return rng.choice(alts)
    return idx


def confuse_syllable(ch: str, rng) -> str:
    """한글 음절 1개를 음운 혼동/받침 탈락 중 하나로 변형."""
    code = ord(ch) - 0xAC00
    cho, jung, jong = code // 588, (code % 588) // 28, code % 28
    mode = rng.choice(["cho", "jung", "jong", "drop"])
    if mode == "cho":
        cho = _pick(cho, _CHO_GROUPS, rng)
    elif mode == "jung":
        jung = _pick(jung, _JUNG_GROUPS, rng)
    elif mode == "jong" and jong:
        jong = _pick(jong, _JONG_GROUPS, rng)
    elif mode == "drop":
        jong = 0  # 받침 탈락(멈춤→머춤).
    return chr(0xAC00 + (cho * 21 + jung) * 28 + jong)


def phonetic_noise(text: str, rng=_random, k: int | None = None) -> str:
    """발화의 한글 1~3글자를 음소 혼동시킨다. STT 오류가 여러 글자에 걸칠 수 있어 글자 수에
    비례해 k 를 정한다(미지정 시)."""
    idxs = [i for i, c in enumerate(text) if "가" <= c <= "힣"]
    if not idxs:
        return text
    if k is None:
        k = 1
        if len(idxs) >= 3 and rng.random() < 0.5:
            k = 2
        if len(idxs) >= 6 and rng.random() < 0.3:
            k = 3
    chars = list(text)
    for i in rng.sample(idxs, min(k, len(idxs))):
        chars[i] = confuse_syllable(chars[i], rng)
    return "".join(chars)
