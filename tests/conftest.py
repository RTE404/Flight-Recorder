import json
import pytest


@pytest.fixture
def fake_llm(monkeypatch):
    """Deterministic fake LiteLLM so recording needs no API key."""
    def _install():
        class _R:
            def __init__(self, c):
                self.choices = [type("C", (), {"message": type("M", (), {"content": c})()})()]

        def completion(model, messages, **kwargs):
            if "sub_questions" in messages[-1]["content"]:
                return _R(json.dumps({"sub_questions": ["qa", "qb"]}))
            return _R("answer-" + str(len(messages[-1]["content"])))

        monkeypatch.setattr("litellm.completion", completion)
    _install()
    return monkeypatch
