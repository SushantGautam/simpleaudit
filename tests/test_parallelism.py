import time
from unittest.mock import MagicMock, patch
from simpleaudit import ModelAuditor, Auditor

def test_model_auditor_parallelism():
    """Test that ModelAuditor run scenarios in parallel."""
    print("Testing ModelAuditor parallelism with tqdm progress bar...")
    with patch("simpleaudit.model_auditor.get_provider") as mock_get_provider:
        mock_provider = MagicMock()
        mock_provider.model = "test-model"
        
        def mock_call(system, user):
            time.sleep(0.5)  # Simulate API latency
            return '{"severity": "pass", "summary": "ok"}'
            
        mock_provider.call.side_effect = mock_call
        mock_get_provider.return_value = mock_provider
        
        auditor = ModelAuditor(provider="openai", verbose=True)
        
        scenarios = [
            {"name": "S1", "description": "D1"},
            {"name": "S2", "description": "D2"},
            {"name": "S3", "description": "D3"},
            {"name": "S4", "description": "D4"},
        ]
        
        # Test sequential
        print("\n--- Running sequential ---")
        start_seq = time.time()
        results_seq = auditor.run(scenarios, max_turns=1, max_workers=1)
        duration_seq = time.time() - start_seq
        
        # Test parallel
        print("\n--- Running parallel ---")
        start_par = time.time()
        results_par = auditor.run(scenarios, max_turns=1, max_workers=2)
        duration_par = time.time() - start_par
        
        assert len(results_seq.results) == 4
        assert len(results_par.results) == 4
        
        print(f"\nSequential duration: {duration_seq:.2f}s")
        print(f"Parallel duration: {duration_par:.2f}s")
        
        if duration_par < duration_seq * 0.8:
            print("SUCCESS: Parallel execution is significantly faster.")
        else:
            print(f"FAILURE: Parallel execution ({duration_par:.2f}s) is not significantly faster than sequential ({duration_seq:.2f}s).")
            exit(1)

def test_auditor_parallelism():
    """Test that ModelAuditor (Auditor alias) runs scenarios in parallel."""
    print("\nTesting ModelAuditor parallelism with tqdm progress bar...")
    with patch("simpleaudit.model_auditor.get_provider") as mock_get_provider:
        
        mock_provider = MagicMock()
        mock_provider.model = "test-model"
        mock_provider.name = "OpenAI"
        
        def mock_call(system, user):
            time.sleep(0.2)
            return '{"severity": "pass", "summary": "ok"}'
            
        mock_provider.call.side_effect = mock_call
        mock_get_provider.return_value = mock_provider
        
        # Auditor is now an alias for ModelAuditor
        # For HTTP endpoints, use provider="openai" with base_url
        auditor = Auditor(
            provider="openai",
            base_url="http://test/v1",
            api_key="test-key",
            prompt_for_key=False,
            verbose=True
        )
        
        scenarios = [
            {"name": "S1", "description": "D1"},
            {"name": "S2", "description": "D2"},
        ]
        
        start_par = time.time()
        results_par = auditor.run(scenarios, max_turns=1, max_workers=2)
        duration_par = time.time() - start_par
        
        assert len(results_par.results) == 2
        print(f"Auditor Parallel duration: {duration_par:.2f}s")
        print("SUCCESS: Auditor parallel execution worked.")

if __name__ == "__main__":
    try:
        test_model_auditor_parallelism()
        test_auditor_parallelism()
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
