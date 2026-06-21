"""BPE 토크나이저 — 한/영 코드스위칭("plate 세트 입고") 대응.

바이트 레벨 BPE 라 한글·영문·숫자·기호를 모두 무손실 처리한다(fai 의 BPE 경험 재활용).
vocab 은 작게(기본 4000) — 도메인이 좁아 작은 어휘로 충분하고 모델·ONNX 가 가벼워진다.
산출물은 artifacts/tokenizer/(gitignore) 에 저장한다.
"""
from __future__ import annotations

from pathlib import Path

from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import TemplateProcessing

DEFAULT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "tokenizer"
SPECIALS = ["<pad>", "<s>", "</s>", "<unk>"]


def train_tokenizer(texts: list[str], out_dir: Path | str = DEFAULT_DIR,
                    vocab_size: int = 4000) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tk = ByteLevelBPETokenizer()
    tk.train_from_iterator(texts, vocab_size=vocab_size, min_frequency=1,
                           special_tokens=SPECIALS)
    tk.save_model(str(out))
    print(f"✅ 토크나이저 학습 — vocab {tk.get_vocab_size()} → {out}")
    return out


def load_tokenizer(out_dir: Path | str = DEFAULT_DIR) -> ByteLevelBPETokenizer:
    out = Path(out_dir)
    tk = ByteLevelBPETokenizer(str(out / "vocab.json"), str(out / "merges.txt"))
    # <s> ... </s> 래핑(인코더 입력 경계).
    tk.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        special_tokens=[("<s>", tk.token_to_id("<s>")), ("</s>", tk.token_to_id("</s>"))],
    )
    return tk


def pad_id(tk: ByteLevelBPETokenizer) -> int:
    return tk.token_to_id("<pad>")
