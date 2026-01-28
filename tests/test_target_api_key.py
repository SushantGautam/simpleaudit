"""
Test for API key handling in ModelAuditor with OpenAI provider.

Run with: pytest tests/test_target_api_key.py
"""

import pytest
from simpleaudit import ModelAuditor


def test_target_api_key_passed_to_provider():
    """Test that api_key is properly passed to the target provider."""
    # Create auditor with an api_key for the target OpenAI provider
    auditor = ModelAuditor(
        provider="openai",
        base_url="http://localhost:8000/v1",
        api_key="test-target-key",
        model="default",
        judge_provider="openai",
        judge_api_key="test-judge-key",
        prompt_for_key=False,
    )
    
    # Verify that the target provider has the API key
    assert auditor.target_provider._api_key == "test-target-key"
    # Verify that the judge provider has its API key
    assert auditor.judge_provider._api_key == "test-judge-key"


def test_target_with_custom_base_url():
    """Test that ModelAuditor works with custom base_url for HTTP endpoints."""
    auditor = ModelAuditor(
        provider="openai",
        base_url="http://localhost:8000/v1",
        api_key="test-key",
        prompt_for_key=False,
    )
    
    # Verify that the base URL is set
    assert auditor.target_provider.base_url == "http://localhost:8000/v1"


def test_same_provider_for_target_and_judge():
    """Test that judge defaults to same provider as target when not specified."""
    auditor = ModelAuditor(
        provider="openai",
        base_url="http://localhost:8000/v1",
        api_key="test-key",
        prompt_for_key=False,
    )
    
    # Should be same instance
    assert auditor.target_provider is auditor.judge_provider
