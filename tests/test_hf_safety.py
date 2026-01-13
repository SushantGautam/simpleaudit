import time
from unittest.mock import MagicMock, patch
from simpleaudit import ModelAuditor, Auditor

def test_huggingface_safety_check():
    """Test that ModelAuditor forces sequential execution for HuggingFace."""
    print("Testing HuggingFace safety check...")
    with patch("simpleaudit.model_auditor.get_provider") as mock_get_provider:
        mock_provider = MagicMock()
        mock_provider.name = "HuggingFace"
        mock_provider.model = "test-hf-model"
        
        def mock_call(system, user):
            time.sleep(0.5)
            return '{"severity": "pass", "summary": "ok"}'
            
        mock_provider.call.side_effect = mock_call
        mock_get_provider.return_value = mock_provider
        
        auditor = ModelAuditor(provider="huggingface", verbose=True)
        
        scenarios = [
            {"name": "S1", "description": "D1"},
            {"name": "S2", "description": "D2"},
        ]
        
        # Even if we request max_workers=2, it should take ~2s (sequential) not ~1s (parallel)
        # 1 scenario = _generate_probe (0.5) + _call_target (0.5) + _judge_conversation (0.5) = 1.5s
        # 2 scenarios = 3s
        start = time.time()
        results = auditor.run(scenarios, max_turns=1, max_workers=2)
        duration = time.time() - start
        
        print(f"HuggingFace duration with max_workers=2: {duration:.2f}s")
        
        # If it was parallel, it would be ~1.5s. If sequential, ~3s.
        assert duration > 2.5
        print("SUCCESS: HuggingFace was forced to sequential execution.")

if __name__ == "__main__":
    try:
        test_huggingface_safety_check()
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
