import sys
from pathlib import Path
# Ensure we use the local simpleaudit package at /home/sushant/D1/simpleaudit
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simpleaudit.model_auditor import ModelAuditor


class MockProvider:
    def __init__(self):
        self.calls = []
        self.model = "mock-model"
        self.name = "Mock"

    def call(self, system, user):
        self.calls.append((system, user))
        return "MOCK_RESPONSE"


def test_system_prompt_values():
    # First test: behavior when bypassing __init__ (existing behavior)
    cases = [
        (None, "expected_system", "None treated as omission (provider receives None)"),
        ("", "expected_system", "Empty string treated as omission (provider receives None)"),
        ("You are helpful.", "expected_system", "String prompt passed through"),
        ([], "expected_system", "Empty list treated as omission (provider receives None)"),
    ]

    for value, user_msg, desc in cases:
        auditor = object.__new__(ModelAuditor)
        auditor.max_turns = 5
        auditor.verbose = False
        auditor._log_lock = None
        auditor._is_parallel = False
        auditor.system_prompt = value
        auditor.target_provider = MockProvider()
        auditor.judge_provider = MockProvider()

        resp = auditor._call_target(user_msg, [])
        passed_system = auditor.target_provider.calls[-1][0]
        print(f"case: {desc!r}, system sent: {passed_system!r}")

    # Second test: enforce constructor validation with monkeypatch of get_provider
    import importlib
    ma = importlib.import_module(ModelAuditor.__module__)

    saved_get = ma.get_provider
    try:
        # Replace get_provider with a stub that returns a MockProvider
        ma.get_provider = lambda *args, **kwargs: MockProvider()

        # These should work
        a1 = ModelAuditor(provider="mock", system_prompt=None)
        a2 = ModelAuditor(provider="mock", system_prompt="hello")

        # When provider has no default, None should result in None (no system)
        print(f"a1.system_prompt: {a1.system_prompt!r}, a2.system_prompt: {a2.system_prompt!r}")

        # Now make the provider return a default system prompt and verify it's used
        class DefaultMock(MockProvider):
            @property
            def default_system_prompt(self):
                return "DEFAULT_PROMPT"

        ma.get_provider = lambda *args, **kwargs: DefaultMock()
        b = ModelAuditor(provider="mock", system_prompt=None)
        print(f"When provider has default, auditor.system_prompt: {b.system_prompt!r}")

        # These should raise TypeError
        for bad in (["a"], {"k": 1}, 123, True):
            try:
                ModelAuditor(provider="mock", system_prompt=bad)
                print(f"ERROR: constructor accepted bad system_prompt: {bad!r}")
            except TypeError as e:
                print(f"Constructor rejected {bad!r} as expected: {e}")
    finally:
        ma.get_provider = saved_get


if __name__ == "__main__":
    test_system_prompt_values()
