"""Backbone LCM 추론 런타임 — holdout_eval 이 from-scratch 와 *같은 인터페이스* 로 비교.

predict_one(text)->{route, semantic_type, answer_intent, confidence, intent_all} 를 제공한다
(infer.LcmRuntime._classify_one 과 동일 형식). 라우팅 로직(_route)도 infer.py 와 동일.
"""
from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoTokenizer

from .backbone_train import BACKBONE, CKPT, BackboneLcm
from .model import action_confidence, predict_heads
from .schema import LabelSpace, decode_intent, load_ssot

DEFAULT_THRESHOLD = 0.8


class BackboneRuntime:
    def __init__(self, ckpt: Path | str = CKPT, threshold: float = DEFAULT_THRESHOLD,
                 max_len: int = 48):
        self.ls = LabelSpace(load_ssot())
        self.tk = AutoTokenizer.from_pretrained(BACKBONE)
        blob = torch.load(ckpt, map_location="cpu")
        self.model = BackboneLcm(self.ls)
        self.model.load_state_dict(blob["model"])
        self.model.eval()
        self.head_specs = self.ls.heads()
        self.threshold = threshold
        self.max_len = max_len

    @torch.no_grad()
    def predict(self, text: str):
        enc = self.tk(text, return_tensors="pt", truncation=True, max_length=self.max_len)
        logits = self.model(enc["input_ids"], enc["attention_mask"])
        heads = predict_heads(logits, self.head_specs)
        return decode_intent(heads, self.ls), action_confidence(logits)

    def _route(self, st: str, ai, action: str, conf: float) -> str:
        if st == "nonsense":
            return "reject"
        if st == "ambiguous":
            return "clarify"                                  # 맥락지시어 → 되묻기(실행 금지)
        if st == "question":
            return "answer_local" if ai else "cloud"
        if st == "chat":
            return "cloud"
        if action != "unknown" and conf >= self.threshold:
            return "execute"
        return "cloud"

    def _classify_one(self, text: str) -> dict:
        intent, conf = self.predict(text)
        st = intent.get("semantic_type", "command")
        ai = intent.get("answer_intent")
        route = self._route(st, ai, intent["action"], conf)
        return {"route": route, "semantic_type": st, "answer_intent": ai,
                "confidence": conf, "intent_all": intent}
