import json
import os
from flightrec import cli
from flightrec.store import Store


def _fake_llm(monkeypatch):
    class _R:
        def __init__(self, c):
            self.choices = [type("C", (), {"message": type("M", (), {"content": c})()})()]

    def completion(model, messages, **kwargs):
        if "sub_questions" in messages[-1]["content"]:
            return _R(json.dumps({"sub_questions": ["a", "b"]}))
        return _R("ok")

    monkeypatch.setattr("litellm.completion", completion)


def test_record_run_creates_complete_trace(tmp_path, monkeypatch):
    _fake_llm(monkeypatch)
    db = os.path.join(tmp_path, "f.db")
    store = Store(db)
    tid = cli.record_run(store, "What is X?")
    t = store.get_trace(tid)
    assert t.status == "complete"
    assert t.task == "What is X?"
    assert len(store.get_events(tid)) > 0
