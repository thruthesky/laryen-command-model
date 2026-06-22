"""추론 + 3계층 디스패치 — 라리엔 클라가 ONNX 로 할 일을 파이썬으로 미리 검증.

3계층(README 와 동일):
  1차 fast-path : 모호성 0 단순 명령. 라리엔 클라의 FastPathRules(이미 구현)이 0ms 처리.
  2차 SML       : 본 모델. action+슬롯을 분류해 VoiceIntent JSON 을 만든다.
                  confidence(action softmax 최댓값) >= 임계값이고 action != unknown 이면 채택.
  3차 CF Gemini : SML 이 unknown 이거나 confidence < 임계값(모호/복합/처음 보는 발화) → 폴백.

라리엔 통합: 본 파일의 LcmRuntime 와 동치인 추론을 클라가 onnxruntime 으로 수행하고,
classify() 의 반환(layer, intent)을 그대로 따른다 — 'sml' 이면 즉시 실행, 'fallback'
이면 기존 classifyVoiceAssistantRemote(text:...) 로 텍스트를 CF 에 보낸다.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch

# 의미 있는 문자(한/영/숫자) — 없으면(빈·구두점·이모지만) 명령일 수 없어 fallback.
_MEANINGFUL = re.compile(r"[가-힣a-zA-Z0-9]")

# 다중동작 분할 — "착용/물약 + 후속" 연결어미를 종결로 복원해 두 명령으로 나눈다.
# (장비/물약 동작 연결만 — "사냥하고 체력 30%" 같은 hunt 옵션 연결은 제외해 오분할 방지.)
_COMPOUND_CONN = [("착용하고", "착용"), ("장착하고", "장착"), ("입고", "입어"),
                  ("먹고", "먹어"), ("마시고", "마셔"), ("쓰고", "써"), ("빨고", "빨아")]


def split_compound(text: str):
    """다중동작이면 [첫 명령, 나머지], 아니면 None. 연결어미를 종결어미로 복원한다."""
    for conn, restore in _COMPOUND_CONN:
        marker = conn + " "
        i = text.find(marker)
        if i > 0:
            first = (text[:i] + restore).strip()
            rest = text[i + len(marker):].strip()
            if rest:
                return [first, rest]
    return None

from .model import LcmEncoder, action_confidence, predict_heads
from .schema import LabelSpace, decode_intent, load_ssot, to_voice_command_json
from .tokenizer import load_tokenizer

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints" / "lcm.pt"
# 0.8 — calibration(label smoothing)이 좋아 threshold 를 올려도 명령 sml 손실이 거의 없다
# (eval sweep: th=0.8 에서 홀드아웃 OOD fallback 0.93·val 명령 sml 0.98·golden 명령 1.0).
# OOD 오작동(잡담→명령 실행)을 더 막는 안전 임계. iter14 재최적화.
DEFAULT_THRESHOLD = 0.8


class LcmRuntime:
    def __init__(self, ckpt: Path | str = CKPT, threshold: float = DEFAULT_THRESHOLD):
        self.ls = LabelSpace(load_ssot())
        self.tk = load_tokenizer()
        blob = torch.load(ckpt, map_location="cpu")
        c = blob["config"]
        self.model = LcmEncoder(c["vocab_size"], self.ls, pad_id=c["pad_id"],
                                d_model=c["d_model"], n_layers=c["layers"], max_len=c["max_len"])
        self.model.load_state_dict(blob["model"])
        self.model.eval()
        self.threshold = threshold

    @torch.no_grad()
    def predict(self, text: str) -> tuple[dict, float]:
        ids = self.tk.encode(text).ids[:32]
        input_ids = torch.tensor([ids], dtype=torch.long)
        attn = torch.ones_like(input_ids, dtype=torch.bool)
        logits = self.model(input_ids, attn)
        # multi_threshold 0.5 — monsters false positive 는 가중 2x + spurious correlation
        # 제거(데이터)로 잡고, 임계는 표준 0.5 로 둬 약한 monster(Bone 등) true positive 유지.
        heads = predict_heads(logits, self.model.head_specs, multi_threshold=0.5)
        intent = decode_intent(heads, self.ls)
        return intent, action_confidence(logits)

    def _classify_one(self, text: str) -> dict:
        """단일 발화 판정(분할 전)."""
        if not _MEANINGFUL.search(text):
            return {"layer": "fallback", "confidence": 0.0, "intent": None}
        intent, conf = self.predict(text)
        if intent["action"] == "unknown" or conf < self.threshold:
            return {"layer": "fallback", "confidence": conf, "intent": None}
        return {"layer": "sml", "confidence": conf, "intent": intent}

    def classify(self, text: str) -> dict:
        """3계층 중 2차(SML) 판정. 다중동작이면 분할해 actions 배열로 결합한다.
        layer='sml'(command.actions 1~N개) 또는 'fallback'."""
        parts = split_compound(text)
        if parts:
            subs = [self._classify_one(p) for p in parts]
            if all(s["layer"] == "sml" for s in subs):
                # 모든 분절이 명확 → actions 배열로 결합(equip/potion 을 먼저 — 라리엔 실행 순서).
                return {"layer": "sml", "confidence": min(s["confidence"] for s in subs),
                        "command": {"actions": [s["intent"] for s in subs], "say": ""}}
            return {"layer": "fallback", "confidence": 0.0, "intent": None}
        r = self._classify_one(text)
        if r["layer"] == "sml":
            return {"layer": "sml", "confidence": r["confidence"],
                    "command": to_voice_command_json(r["intent"])}
        return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="*", help="발화(생략 시 대화형)")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = ap.parse_args()
    rt = LcmRuntime(threshold=args.threshold)

    def run(t: str):
        import json
        print(json.dumps(rt.classify(t), ensure_ascii=False))

    if args.text:
        run(" ".join(args.text))
    else:
        print("발화를 입력하세요(빈 줄 종료):")
        while True:
            try:
                t = input("> ").strip()
            except EOFError:
                break
            if not t:
                break
            run(t)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
