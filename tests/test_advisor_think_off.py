"""The LLM relevance advisor must force think=False — qwen3.5-mayring (the advisor model)
emits a 4000-token <think> chain on complex relevance prompts (36s vs 1.5s), and the
num_predict cap then truncates mid-reasoning → empty scores + blown hook budget."""
from mayring_core.memory import retrieval
from mayring_core.memory.schema import Chunk


def test_advisor_forces_think_off(monkeypatch):
    captured = {}

    def _fake_generate(url, model, prompt, **kwargs):
        captured.update(kwargs)
        # a valid score map so the parse path runs
        return '{"0": 0.9, "1": 0.2}'

    monkeypatch.setattr("mayring_core.ollama_client.generate", _fake_generate)

    cands = [
        Chunk(chunk_id="chk_0", source_id="s", chunk_level="event", ordinal=0,
              text="reranker training", text_hash="sha256:a", created_at="2026-06-08"),
        Chunk(chunk_id="chk_1", source_id="s", chunk_level="event", ordinal=1,
              text="unrelated", text_hash="sha256:b", created_at="2026-06-08"),
    ]
    scores = retrieval._llm_relevance_scores("reranker", cands, "http://gpu:11434",
                                             model="qwen3.5-mayring:2b")
    assert captured.get("think") is False  # MUST disable thinking
    assert scores.get("chk_0") == 0.9
