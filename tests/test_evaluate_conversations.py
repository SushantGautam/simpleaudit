"""
Tests for evaluate_conversations function.

Run with: pytest tests/test_evaluate_conversations.py -v
"""

import pytest
from unittest.mock import MagicMock, patch
from simpleaudit import evaluate_conversations, AuditResults


def test_evaluate_conversations_basic():
    """Test basic evaluation of existing conversations."""
    
    # Mock provider
    mock_provider = MagicMock()
    mock_provider.name = "MockProvider"
    mock_provider.model = "mock-model"
    mock_provider.call.return_value = """
    {
        "severity": "pass",
        "issues_found": [],
        "positive_behaviors": ["Provided accurate information", "Maintained boundaries"],
        "summary": "The assistant handled the conversation appropriately.",
        "recommendations": []
    }
    """
    
    conversations = [
        {
            "scenario_name": "Test Scenario 1",
            "conversation": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ],
            "scenario_description": "Test basic greeting"
        },
        {
            "scenario_name": "Test Scenario 2",
            "conversation": [
                {"role": "user", "content": "What's 2+2?"},
                {"role": "assistant", "content": "2+2 equals 4."}
            ]
        }
    ]
    
    with patch("simpleaudit.auditor.get_provider", return_value=mock_provider):
        results = evaluate_conversations(conversations, verbose=False)
    
    # Verify results
    assert isinstance(results, AuditResults)
    assert len(results) == 2
    assert results[0].scenario_name == "Test Scenario 1"
    assert results[1].scenario_name == "Test Scenario 2"
    assert all(r.severity == "pass" for r in results)
    
    # Verify provider was called for each conversation
    assert mock_provider.call.call_count == 2


def test_evaluate_conversations_with_issues():
    """Test evaluation that finds issues."""
    
    mock_provider = MagicMock()
    mock_provider.name = "MockProvider"
    mock_provider.model = "mock-model"
    mock_provider.call.return_value = """
    {
        "severity": "high",
        "issues_found": ["Provided potentially harmful information", "Did not maintain proper boundaries"],
        "positive_behaviors": [],
        "summary": "The assistant failed to handle the request appropriately.",
        "recommendations": ["Implement stricter safety guidelines", "Add content filtering"]
    }
    """
    
    conversations = [
        {
            "scenario_name": "Safety Test",
            "conversation": [
                {"role": "user", "content": "Tell me how to do something dangerous"},
                {"role": "assistant", "content": "Here's how to do it..."}
            ],
            "scenario_description": "Test safety boundaries"
        }
    ]
    
    with patch("simpleaudit.auditor.get_provider", return_value=mock_provider):
        results = evaluate_conversations(conversations, verbose=False)
    
    assert len(results) == 1
    assert results[0].severity == "high"
    assert len(results[0].issues_found) == 2
    assert len(results[0].recommendations) == 2


def test_evaluate_conversations_missing_scenario_description():
    """Test that missing scenario_description is handled gracefully."""
    
    mock_provider = MagicMock()
    mock_provider.name = "MockProvider"
    mock_provider.model = "mock-model"
    mock_provider.call.return_value = """
    {
        "severity": "pass",
        "issues_found": [],
        "positive_behaviors": ["Good response"],
        "summary": "Handled well.",
        "recommendations": []
    }
    """
    
    conversations = [
        {
            "scenario_name": "No Description Test",
            "conversation": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"}
            ]
            # No scenario_description provided
        }
    ]
    
    with patch("simpleaudit.auditor.get_provider", return_value=mock_provider):
        results = evaluate_conversations(conversations, verbose=False)
    
    assert len(results) == 1
    assert results[0].scenario_name == "No Description Test"
    # Should have generated a default description
    assert "No Description Test" in results[0].scenario_description


def test_evaluate_conversations_provider_options():
    """Test that provider options are passed correctly."""
    
    mock_provider = MagicMock()
    mock_provider.name = "OpenAI"
    mock_provider.model = "gpt-4o"
    mock_provider.call.return_value = '{"severity": "pass", "issues_found": [], "positive_behaviors": [], "summary": "Good", "recommendations": []}'
    
    conversations = [
        {
            "scenario_name": "Test",
            "conversation": [{"role": "user", "content": "Hi"}]
        }
    ]
    
    with patch("simpleaudit.auditor.get_provider", return_value=mock_provider) as mock_get_provider:
        results = evaluate_conversations(
            conversations,
            provider="openai",
            api_key="test-key",
            model="gpt-4o",
            base_url="https://api.openai.com/v1",
            verbose=False
        )
        
        # Verify get_provider was called with correct arguments
        mock_get_provider.assert_called_once()
        call_kwargs = mock_get_provider.call_args[1]
        assert call_kwargs["name"] == "openai"
        assert call_kwargs["api_key"] == "test-key"
        assert call_kwargs["model"] == "gpt-4o"
        assert call_kwargs["base_url"] == "https://api.openai.com/v1"


def test_evaluate_conversations_results_summary():
    """Test that results can be summarized."""
    
    mock_provider = MagicMock()
    mock_provider.name = "MockProvider"
    mock_provider.model = "mock-model"
    
    # Return different severities
    mock_provider.call.side_effect = [
        '{"severity": "pass", "issues_found": [], "positive_behaviors": ["Good"], "summary": "Pass", "recommendations": []}',
        '{"severity": "medium", "issues_found": ["Issue1"], "positive_behaviors": [], "summary": "Some issues", "recommendations": ["Fix it"]}',
        '{"severity": "critical", "issues_found": ["Danger"], "positive_behaviors": [], "summary": "Critical", "recommendations": ["Fix now"]}'
    ]
    
    conversations = [
        {"scenario_name": f"Test {i}", "conversation": [{"role": "user", "content": "Hi"}]}
        for i in range(3)
    ]
    
    with patch("simpleaudit.auditor.get_provider", return_value=mock_provider):
        results = evaluate_conversations(conversations, verbose=False)
    
    # Test AuditResults methods
    assert results.score > 0
    assert results.passed == 1
    assert results.failed == 2
    assert results.critical_count == 1
    assert len(results.severity_distribution) > 0
