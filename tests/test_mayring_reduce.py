from mayring_core.memory.ingestion.mayring_process import reduce_prompt


def test_reduce_prompt_mode_deductive_forbids_new():
    p = reduce_prompt("auth code", "security", ["auth", "api"], mode="deductive")
    assert "security" in p
    assert "auth, api" in p
    assert "keine neue" in p.lower() or "nur aus" in p.lower()


def test_reduce_prompt_mode_inductive_no_anchor_pressure():
    p = reduce_prompt("auth code", "security", ["auth"], mode="inductive")
    assert "frei" in p.lower() or "neu" in p.lower()


def test_reduce_prompt_default_is_bare_label():
    p = reduce_prompt("auth code", "security")
    assert "NUR mit dem finalen Kategorie-Label" in p


def test_reduce_prompt_structured_asks_json():
    p = reduce_prompt("auth code", "security", structured=True)
    assert "paraphrase" in p and "generalization" in p and "label" in p
    assert "JSON" in p
