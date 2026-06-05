"""_default_generate muss `think` an ollama_client.generate durchreichen.

WHY(2026-06-05): qwen3.5:2b (Thinking-Modell) verbrauchte ohne think=False das
num_predict-Budget mit dem Thinking-Trace, bevor das JSON-Label kam → leeres Label
(done_reason=length). Die Reduktion setzt think=False; ohne diesen Passthrough
warf _default_generate TypeError (kein think-Param)."""
from __future__ import annotations

from mayring_core import providers


def test_default_generate_forwards_think(monkeypatch):
    captured: dict = {}

    def _fake_generate(host, model, prompt, **kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr("mayring_core.ollama_client.generate", _fake_generate)
    out = providers._default_generate(
        "p", "http://x", "qwen3.5:2b", "label", think=False)
    assert out == "ok"
    assert captured["think"] is False


def test_default_generate_think_defaults_none(monkeypatch):
    captured: dict = {}

    def _fake_generate(host, model, prompt, **kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr("mayring_core.ollama_client.generate", _fake_generate)
    providers._default_generate("p", "http://x", "mistral:7b-instruct", "label")
    assert captured["think"] is None
