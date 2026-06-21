"""bpe_ref(순수 파이썬) ↔ HF tokenizers parity — dart 포팅의 정확성 보장.

bpe_ref 가 학습에 쓴 토크나이저와 *완전히 같은* 토큰 id 를 내야, dart 로 옮긴 인코더로
ONNX 추론이 맞다. 한/영/혼용/숫자/기호 다양한 입력에서 1:1 일치를 강제한다.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lcm.bpe_ref import BpeRef  # noqa: E402
from lcm.tokenizer import DEFAULT_DIR  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (DEFAULT_DIR / "vocab.json").exists(),
    reason="토크나이저 없음(python -m lcm.train 먼저)")

TEXTS = [
    "강남에서 사냥", "왼쪽으로 가", "멈춰", "강철 세트 착용", "인벤토리 열어",
    "체력 물약 먹어", "안녕 너 누구야", "plate 세트 입고 사냥",     # 한/영 혼용
    "5시 방향으로 걸어가", "강동 꽃밭에서 Bone 사냥하고 체력 30% 피신",  # 숫자/%
    "hello", "auto hunt on", "what is a caster", "go to gangnam",     # 영어
    "도착하면 자동공격!!", "물약, 물약, 물약",                          # 기호/반복
]


@pytest.fixture(scope="module")
def ref_and_hf():
    from lcm.tokenizer import load_tokenizer
    return BpeRef.load(), load_tokenizer()


def test_bpe_parity_with_special(ref_and_hf):
    ref, hf = ref_and_hf
    for t in TEXTS:
        got = ref.encode_with_special(t)
        want = hf.encode(t).ids
        assert got == want, f"'{t}'\n  ref ={got}\n  hf  ={want}"


def test_bpe_handles_empty_and_space(ref_and_hf):
    ref, hf = ref_and_hf
    for t in ("", " ", "  멈춰  "):
        assert ref.encode_with_special(t) == hf.encode(t).ids, f"'{t}'"
