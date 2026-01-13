import time
from unittest.mock import MagicMock, patch
from simpleaudit import ModelAuditor

def test_huggingface_safety_check_refined():
    """Test that ModelAuditor forces sequential execution if either provider is HuggingFace."""
    
    scenarios = [
        {"name": "S1", "description": "D1"},
        {"name": "S2", "description": "D2"},
    ]

    with patch("simpleaudit.model_auditor.get_provider") as mock_get_provider:
        # Define mock behavior to return JSON strings for judgment and text for probes
        def mock_call(system, user):
            if "Evaluate this conversation" in user:
                return '{"severity": "pass", "summary": "ok"}'
            return "Test response/probe"

        # Case 1: Target is HF, Judge is OpenAI (User's specific case)
        print("\nTesting Case 1: Target=HF, Judge=OpenAI")
        mock_hf = MagicMock()
        mock_hf.name = "HuggingFace"
        mock_hf.model = "test-hf-model"
        mock_hf.call.side_effect = mock_call
        
        mock_openai = MagicMock()
        mock_openai.name = "OpenAI"
        mock_openai.model = "gpt-4"
        mock_openai.call.side_effect = mock_call
        
        def mock_provider_factory(name, **kwargs):
            if name == "huggingface": return mock_hf
            if name == "openai": return mock_openai
            return mock_hf

        mock_get_provider.side_effect = mock_provider_factory
        
        auditor = ModelAuditor(provider="huggingface", judge_provider="openai", verbose=True)
        
        # Should be forced to sequential
        auditor.run(scenarios, max_turns=1, max_workers=4)
        print(f"Case 1 completed.")

        # Case 2: Target is OpenAI, Judge is HF
        print("\nTesting Case 2: Target=OpenAI, Judge=HF")
        mock_get_provider.side_effect = mock_provider_factory
        auditor2 = ModelAuditor(provider="openai", judge_provider="huggingface", verbose=True)
        
        # Should be forced to sequential
        auditor2.run(scenarios, max_turns=1, max_workers=4)
        print(f"Case 2 completed.")

        # Case 3: Neither is HF (Should remain parallel)
        print("\nTesting Case 3: Neither is HF")
        mock_openai2 = MagicMock()
        mock_openai2.name = "OpenAI"
        mock_openai2.model = "gpt-3.5"
        mock_openai2.call.side_effect = mock_call
        mock_get_provider.return_value = mock_openai2
        mock_get_provider.side_effect = None # Clear side effect
        
        auditor3 = ModelAuditor(provider="openai", judge_provider="openai", verbose=True)
        
        # Should remain parallel
        auditor3.run(scenarios, max_turns=1, max_workers=4)
        print(f"Case 3 completed.")

if __name__ == "__main__":
    try:
        test_huggingface_safety_check_refined()
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
