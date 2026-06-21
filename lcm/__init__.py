"""Laryen Command Model (LCM) — 라리엔 명령어 이해 SML.

발화 텍스트 → VoiceIntent JSON 분류(생성 아님). 라리엔 클라가 sherpa_onnx 와 같은
onnxruntime 으로 온디바이스 추론한다. 상세는 README.md 와 schema.py 참조.
"""

__version__ = "0.1.0"
