"""jsonl (text, intent) → 텐서. encode_intent 로 헤드별 라벨을 만든다."""
from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .schema import LabelSpace, encode_intent


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
        """공백 변형(제거/중복) — STT 전사·사용자 입력의 띄어쓰기 불규칙에 강건하게."""
        r = random.random()
        if r < self.aug_p * 0.6:
            return text.replace(" ", "")           # 공백 전부 제거
        if r < self.aug_p:
            return text.replace(" ", "  ")          # 공백 중복
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
