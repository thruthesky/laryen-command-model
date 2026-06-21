"""jsonl (text, intent) → 텐서. encode_intent 로 헤드별 라벨을 만든다."""
from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .schema import LabelSpace, encode_intent


def _jamo_one(ch: str) -> str:
    code = ord(ch) - 0xAC00
    cho, jung, jong = code // 588, (code % 588) // 28, code % 28
    if jong and random.random() < 0.5:
        jong = 0                                    # 받침 탈락(멈춤→머춤)
    else:
        jung = (jung + random.choice([-1, 1])) % 21  # 모음 혼동(사냥→사녕)
    return chr(0xAC00 + (cho * 21 + jung) * 28 + jong)


def _jamo_noise(text: str) -> str:
    """한글 1~2글자의 받침 탈락/모음 ±1 변형(STT 음소 오류 모사 — 더 강건하게)."""
    idxs = [i for i, c in enumerate(text) if "가" <= c <= "힣"]
    if not idxs:
        return text
    k = 2 if len(idxs) >= 4 and random.random() < 0.4 else 1
    chars = list(text)
    for i in random.sample(idxs, k=min(k, len(idxs))):
        chars[i] = _jamo_one(chars[i])
    return "".join(chars)


def read_jsonl(path: Path | str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class LcmDataset(Dataset):
    def __init__(self, jsonl_path: Path | str, tokenizer, ls: LabelSpace,
                 max_len: int = 32, augment: bool = False, aug_p: float = 0.3):
        self.rows = read_jsonl(jsonl_path)
        self.tk = tokenizer
        self.ls = ls
        self.max_len = max_len
        self.augment = augment  # train 에서만 — STT/입력 공백 불규칙 대응
        self.aug_p = aug_p
        self.head_specs = ls.heads()  # (name, kind, labels)

    def __len__(self) -> int:
        return len(self.rows)

    def _augment_text(self, text: str) -> str:
        """STT/입력 노이즈 모사 — 공백 변형 + 자모 변형(받침 탈락·모음 혼동).
        sherpa STT 는 음소 단위라 "멈춰"→"멈처"·"사냥"→"사양" 류 자모 오류가 실전 노이즈."""
        r = random.random()
        if r < self.aug_p * 0.18:
            return text.replace(" ", "")           # 공백 전부 제거
        if r < self.aug_p * 0.3:
            return text.replace(" ", "  ")          # 공백 중복
        if r < self.aug_p:
            return _jamo_noise(text)                # 자모 변형(받침/모음 — 비중 70%)
        return text

    def __getitem__(self, i: int) -> dict:
        row = self.rows[i]
        text = self._augment_text(row["text"]) if self.augment else row["text"]
        ids = self.tk.encode(text).ids[: self.max_len]
        labels = encode_intent(row["intent"], self.ls)
        return {"input_ids": ids, "labels": labels}


def make_collate(pad_id: int, ls: LabelSpace, max_len: int = 32):
    head_specs = ls.heads()

    def collate(batch: list[dict]) -> dict:
        n = len(batch)
        lengths = [len(b["input_ids"]) for b in batch]
        L = min(max(lengths), max_len)
        input_ids = torch.full((n, L), pad_id, dtype=torch.long)
        attn = torch.zeros((n, L), dtype=torch.bool)
        for r, b in enumerate(batch):
            ids = b["input_ids"][:L]
            input_ids[r, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attn[r, : len(ids)] = True
        labels: dict[str, torch.Tensor] = {}
        for name, kind, _ in head_specs:
            if kind in ("single",):
                labels[name] = torch.tensor([b["labels"][name] for b in batch], dtype=torch.long)
            elif kind == "binary":
                labels[name] = torch.tensor([b["labels"][name] for b in batch], dtype=torch.float)
            else:  # multi
                labels[name] = torch.tensor([b["labels"][name] for b in batch], dtype=torch.float)
        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}

    return collate
