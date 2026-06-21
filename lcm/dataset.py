"""jsonl (text, intent) → 텐서. encode_intent 로 헤드별 라벨을 만든다."""
from __future__ import annotations

import json
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
    def __init__(self, jsonl_path: Path | str, tokenizer, ls: LabelSpace, max_len: int = 32):
        self.rows = read_jsonl(jsonl_path)
        self.tk = tokenizer
        self.ls = ls
        self.max_len = max_len
        self.head_specs = ls.heads()  # (name, kind, labels)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        row = self.rows[i]
        ids = self.tk.encode(row["text"]).ids[: self.max_len]
        labels = encode_intent(row["intent"], self.ls)
        item = {"input_ids": ids, "labels": labels}
        return item


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
