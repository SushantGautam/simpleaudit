from unittest.mock import patch, MagicMock
from simpleaudit import ModelAuditor

def test_model_auditor_base_url():
    print("Testing ModelAuditor with custom base_url...")
    
    # Patch get_provider WHERE IT IS USED (in model_auditor.py)
    with patch("simpleaudit.model_auditor.get_provider") as mock_get_provider:
        # Configure mock to return something with a .model attribute
        mock_provider = MagicMock()
        mock_provider.model = "test-model"
        mock_get_provider.return_value = mock_provider
        
        auditor = ModelAuditor(
            provider="openai",
            model="custom-model",
            base_url="https://custom.api.com/v1",
            judge_provider="openai",
            judge_base_url="https://judge.api.com/v1"
        )
        
        # Verify target provider call
        args, kwargs = mock_get_provider.call_args_list[0]
        print(f"Target provider kwargs: {kwargs}")
        assert kwargs["base_url"] == "https://custom.api.com/v1"
        assert kwargs["name"] == "openai"
        
        # Verify judge provider call
        args, kwargs = mock_get_provider.call_args_list[1]
        print(f"Judge provider kwargs: {kwargs}")
        assert kwargs["base_url"] == "https://judge.api.com/v1"
        assert kwargs["name"] == "openai"
        
        print("SUCCESS: base_url correctly passed to providers.")

if __name__ == "__main__":
    test_model_auditor_base_url()
