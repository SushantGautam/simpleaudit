import json
import pytest
from simpleaudit.auditor import evaluate_conversations


def make_conversations(n=3):
    return [
        {
            "scenario_name": f"S{i}",
            "conversation": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ],
        }
        for i in range(n)
    ]


class FakeProvider:
    def __init__(self, name="Mock", model="mock-model", severity="pass"):
        self.name = name
        self.model = model
        self.call_count = 0
        self._severity = severity

    def call(self, system, user, extra_body=None):
        # Simple deterministic JSON response; accept optional extra_body (guided schema)
        self.call_count += 1
        return json.dumps({
            "severity": self._severity,
            "issues_found": [],
            "positive_behaviors": ["handled appropriately"],
            "summary": "Automated judgment",
            "recommendations": []
        })


def test_evaluate_conversations_sequential(monkeypatch):
    convs = make_conversations(2)
    fake = FakeProvider()

    # Patch get_provider to return our fake provider
    monkeypatch.setattr("simpleaudit.auditor.get_provider", lambda **kwargs: fake)

    results = evaluate_conversations(convs, verbose=False, max_workers=1)

    assert len(results) == 2
    assert results.passed == 2
    assert results.severity_distribution == {"pass": 2}
    # call_count should be equal to number of conversations
    assert fake.call_count == 2


def test_evaluate_conversations_parallel(monkeypatch):
    convs = make_conversations(5)
    fake = FakeProvider()

    monkeypatch.setattr("simpleaudit.auditor.get_provider", lambda **kwargs: fake)

    results = evaluate_conversations(convs, verbose=False, max_workers=3)

    assert len(results) == 5
    assert results.passed == 5
    assert results.severity_distribution == {"pass": 5}
    # ensure the shared provider was called for each conversation
    assert fake.call_count == 5


def test_huggingface_fallback(monkeypatch, capsys):
    convs = make_conversations(2)
    # provider that mimics HuggingFace local runner
    fake = FakeProvider(name="HuggingFace")
    monkeypatch.setattr("simpleaudit.auditor.get_provider", lambda **kwargs: fake)

    # request parallel execution, but HF should force single-threaded
    results = evaluate_conversations(convs, verbose=True, max_workers=4)

    captured = capsys.readouterr()
    assert "HuggingFace judge provider is not thread-safe" in captured.out
    assert len(results) == 2
    assert fake.call_count == 2
