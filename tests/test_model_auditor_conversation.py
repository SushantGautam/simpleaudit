from unittest.mock import patch

from simpleaudit.model_auditor import ModelAuditor
from simpleaudit.results import AuditResults


class FakeProvider:
    def __init__(self, name="Fake", model="fake-model"):
        self.model = model
        self._name = name

    @property
    def name(self):
        return self._name

    def call(self, system: str, user: str, extra_body=None) -> str:
        # Echo the user prompt so it's easy to assert
        if system and "EVALUATE" in system.upper():
            # Return a valid JSON judgment for the judge calls
            return '{"severity": "pass", "issues_found": [], "positive_behaviors": [], "summary": "OK", "recommendations": []}'
        return f"Reply to: {user}"


def test_model_auditor_records_user_and_assistant_roles():
    # Use patch to inject fake providers
    with patch("simpleaudit.model_auditor.get_provider") as mock_get_provider:
        fake_target = FakeProvider(name="target")
        fake_judge = FakeProvider(name="judge")

        # First call returns target, second returns judge (if judge_provider is set)
        mock_get_provider.return_value = fake_target

        auditor = ModelAuditor(provider="anthropic", model="fake-model", verbose=False)

        # Override judge with separate fake if they're different objects
        if auditor.judge_provider is not auditor.target_provider:
            auditor.judge_provider = fake_judge
        else:
            auditor.judge_provider = fake_judge

        # Run a single simple scenario with 2 turns
        results = auditor.run([{"name": "Test", "description": "desc"}], max_turns=2, max_workers=1)

        assert isinstance(results, AuditResults)
        assert len(results) == 1

        conv = results[0].conversation
        # Expect 2 turns -> 4 messages (user, assistant) x2
        assert len(conv) == 4

        # Check alternating roles
        assert conv[0]["role"] == "user"
        assert conv[1]["role"] == "assistant"
        assert conv[2]["role"] == "user"
        assert conv[3]["role"] == "assistant"

        # Check contents were echoed
        assert "Reply to:" in conv[1]["content"]
        assert "Reply to:" in conv[3]["content"]
