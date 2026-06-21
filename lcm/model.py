"""LcmEncoder — 초경량 인코더 + 멀티헤드 분류기(생성 디코더 없음).

작은 Transformer 인코더로 발화를 풀링한 뒤, schema.LabelSpace.heads() 의 각 헤드를
linear 로 분류한다. action 은 항상, 나머지 슬롯은 해당 action 일 때만 의미가 있으나
(비활성 슬롯의 타겟은 <none>/0), 학습은 모든 헤드에 loss 를 적용한다 — <none> 예측이
곧 "이 슬롯 비활성" 이라 마스킹 없이도 자연 학습된다.

기본 d_model=128 / layer 2 / head 4 → 파라미터 수십만, int8 ONNX 수 MB. M5 에서 분 단위 학습.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .schema import LabelSpace


class LcmEncoder(nn.Module):
    def __init__(self, vocab_size: int, ls: LabelSpace, pad_id: int = 0,
                 d_model: int = 128, n_layers: int = 2, n_heads: int = 4,
                 d_ff: int = 256, max_len: int = 64, dropout: float = 0.1):
        super().__init__()
        self.pad_id = pad_id
        self.ls = ls
        self.tok = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_ff, dropout=dropout,
            batch_first=True, activation="gelu")
        # enable_nested_tensor=False: MPS 는 nested tensor 최적화 경로
        # (_nested_tensor_from_mask_left_aligned)가 미구현이라 끈다(작은 모델이라 영향 0).
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)

        self.head_specs = ls.heads()
        self.heads = nn.ModuleDict({
            name: nn.Linear(d_model, len(labels) if kind != "binary" else 1)
            for name, kind, labels in self.head_specs
        })
        self.max_len = max_len

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device).clamp_max(self.max_len - 1)
        x = self.tok(input_ids) + self.pos(pos)[None, :, :]
        # additive float padding mask(-1e9) — boolean src_key_padding_mask 는 trace 시
        # 상수 폴딩되어 ONNX 가 export 입력 길이에 고정되는 회귀가 있다(다른 길이 입력에서
        # 출력이 틀어짐). float mask 는 텐서 연산으로 남아 어떤 길이에도 일반화된다.
        neg = torch.zeros(input_ids.shape, dtype=x.dtype, device=x.device)
        neg = neg.masked_fill(~attention_mask, -1e9)
        h = self.encoder(x, src_key_padding_mask=neg)
        h = self.norm(h)
        # masked mean pooling.
        m = attention_mask.unsqueeze(-1).to(x.dtype)
        pooled = (h * m).sum(1) / m.sum(1).clamp_min(1.0)
        return pooled

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict:
        pooled = self.encode(input_ids, attention_mask)
        out = {}
        for name, kind, _ in self.head_specs:
            logits = self.heads[name](pooled)
            out[name] = logits.squeeze(-1) if kind == "binary" else logits
        return out


def multihead_loss(logits: dict, labels: dict, head_specs) -> torch.Tensor:
    ce = nn.functional.cross_entropy
    bce = nn.functional.binary_cross_entropy_with_logits
    total = logits["action"].new_zeros(())
    for name, kind, _ in head_specs:
        if kind == "single":
            # action 헤드를 더 강하게(가장 중요한 라우팅 결정). label smoothing 으로
            # OOD 과신을 억제한다(softmax 가 학습 분포 밖 입력을 1.0 으로 확신 → fallback
            # 안전장치 무력화하는 문제 완화).
            w = 2.0 if name == "action" else 1.0
            ls_eps = 0.1 if name == "action" else 0.0
            total = total + w * ce(logits[name], labels[name], label_smoothing=ls_eps)
        elif kind == "binary":
            total = total + bce(logits[name], labels[name])
        else:  # multi — monsters 는 32종 중 1~2개만 1 인 희소 멀티라벨이라(대부분 0)
            # unknown 표본이 많으면 "전부 0" 편향. 가중으로 신호를 키우되 2x(3x 는 false
            # positive 유발 — 약한 logit 이 임계를 넘어 미언급 monster 삽입).
            w = 2.0 if name == "monsters" else 1.0
            total = total + w * bce(logits[name], labels[name])
    return total


@torch.no_grad()
def predict_heads(logits: dict, head_specs, multi_threshold: float = 0.5) -> dict:
    """logits → 헤드별 예측 인덱스(single/binary 정수, multi 0/1 리스트)."""
    out = {}
    for name, kind, _ in head_specs:
        if kind == "single":
            out[name] = int(logits[name].argmax(-1).item())
        elif kind == "binary":
            out[name] = int((torch.sigmoid(logits[name]) > 0.5).item())
        else:
            probs = torch.sigmoid(logits[name]).squeeze(0)
            out[name] = [int(p > multi_threshold) for p in probs]
    return out


@torch.no_grad()
def action_confidence(logits: dict) -> float:
    """action 헤드 softmax 최댓값 = 라우팅 신뢰도(3계층 폴백 임계값에 사용)."""
    return float(torch.softmax(logits["action"], -1).max().item())
