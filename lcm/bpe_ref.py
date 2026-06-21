"""순수 파이썬 ByteLevelBPE 인코더 — dart 1:1 포팅 레퍼런스.

**왜 필요한가**: 라리엔 클라가 onnxruntime 으로 LCM 을 추론하려면 발화를 *모델과 동일한*
토큰 id 로 바꿔야 한다. 학습에 쓴 HF `tokenizers`(Rust)는 Flutter 에 없으므로 **dart 로
같은 BPE 를 재현**해야 한다(INTEGRATION.md §BPE). 이 파일은 외부 BPE 라이브러리 없이
vocab.json/merges.txt 만으로 인코딩하는 *순수 로직* 이라, 그대로 dart 로 옮길 수 있다.
`tests/test_bpe.py` 가 HF tokenizers 와 출력이 1:1 같음을 보장한다(parity).

알고리즘(GPT-2 ByteLevelBPE):
  1) (add_prefix_space) 텍스트 앞에 공백 1개.
  2) GPT-2 정규식으로 pre-tokenize(단어/숫자/기호/공백 조각).
  3) 각 조각을 UTF-8 byte → byte_to_unicode 매핑(0~255 → 가시 유니코드).
  4) merges.txt 순위대로 인접 쌍 병합(BPE).
  5) 각 subword → vocab id. (<s> ... </s> 래핑은 호출부에서)
"""
from __future__ import annotations

import json
from pathlib import Path

import regex  # \p{L} 등 유니코드 속성 — dart 는 RegExp(unicode:true) 로 대응

DEFAULT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "tokenizer"

# GPT-2 ByteLevel pre-tokenize 정규식(HF ByteLevel 기본과 동일).
_PAT = regex.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


def bytes_to_unicode() -> dict[int, str]:
    """0~255 byte → 가시 유니코드 char(GPT-2 표준). dart 도 동일 표를 만든다."""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


class BpeRef:
    def __init__(self, vocab: dict[str, int], merges: list[str],
                 add_prefix_space: bool = False):  # HF ByteLevelBPETokenizer 기본과 일치
        self.encoder = vocab
        self.bpe_ranks = {tuple(m.split()): i for i, m in enumerate(merges)}
        self.byte_encoder = bytes_to_unicode()
        self.add_prefix_space = add_prefix_space

    @staticmethod
    def load(dir_: Path | str = DEFAULT_DIR) -> "BpeRef":
        d = Path(dir_)
        vocab = json.loads((d / "vocab.json").read_text(encoding="utf-8"))
        lines = (d / "merges.txt").read_text(encoding="utf-8").splitlines()
        merges = [ln for ln in lines if ln and not ln.startswith("#")]
        return BpeRef(vocab, merges)

    def _bpe(self, token: str) -> list[str]:
        word = tuple(token)
        if len(word) < 2:
            return list(word)
        while True:
            pairs = set(zip(word, word[1:]))
            best = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if best not in self.bpe_ranks:
                break
            first, second = best
            new: list[str] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new.append(first + second)
                    i += 2
                else:
                    new.append(word[i])
                    i += 1
            word = tuple(new)
            if len(word) == 1:
                break
        return list(word)

    def encode(self, text: str, unk: str = "<unk>") -> list[int]:
        """텍스트 → 토큰 id(특수토큰 <s>/</s> 미포함 — 호출부에서 래핑)."""
        if self.add_prefix_space and text and not text[0].isspace():
            text = " " + text
        ids: list[int] = []
        for piece in _PAT.findall(text):
            tok = "".join(self.byte_encoder[b] for b in piece.encode("utf-8"))
            for sub in self._bpe(tok):
                ids.append(self.encoder.get(sub, self.encoder.get(unk, 3)))
        return ids

    def encode_with_special(self, text: str) -> list[int]:
        """<s> ... </s> 래핑(HF post_processor 와 동일 — 모델 입력 형식)."""
        bos = self.encoder.get("<s>", 1)
        eos = self.encoder.get("</s>", 2)
        return [bos] + self.encode(text) + [eos]
