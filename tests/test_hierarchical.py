import time
import threading
from simpleaudit import ModelAuditor
from unittest.mock import MagicMock, patch

def test_hierarchical_progress():
    scenarios = [
        {"name": f"Scenario {i}", "description": f"Testing scenario {i}"}
        for i in range(4)
    ]
    
    with patch("simpleaudit.model_auditor.get_provider") as mock_get_provider:
        def mock_call(system, user):
            time.sleep(0.5) # Simulate latency
            return "Response content"
        
        mock_provider = MagicMock()
        mock_provider.name = "Mock"
        mock_provider.model = "test-model"
        mock_provider.call.side_effect = mock_call
        mock_get_provider.return_value = mock_provider
        
        auditor = ModelAuditor(provider="openai", verbose=True)
        
        print("\n--- Testing Hierarchical Progress (4 workers) ---")
        # In this mode, we should see 4 active bars + 1 global bar
        results = auditor.run(scenarios, max_turns=3, max_workers=4)
        print("\nAudit Complete")

if __name__ == "__main__":
    test_hierarchical_progress()
