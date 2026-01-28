"""
SimpleAudit - Lightweight AI Safety Auditing Framework

A simple, effective tool for red-teaming AI systems using LLMs as auditor and judge.

Supports multiple providers:
- Anthropic (Claude) - default
- OpenAI (GPT-4, GPT-5, etc.) - supports custom base_url for OpenAI-compatible endpoints
- Grok (xAI)
- HuggingFace (local transformers)
- Ollama (local models)

Usage:
    # Audit HTTP endpoint (use ModelAuditor with OpenAI provider)
    from simpleaudit import ModelAuditor
    auditor = ModelAuditor(
        provider="openai",
        base_url="http://localhost:8000/v1",
        model="default"
    )
    results = auditor.run("safety")
    
    # Audit model directly via API
    from simpleaudit import ModelAuditor
    auditor = ModelAuditor(provider="anthropic", system_prompt="You are helpful.")
    results = auditor.run("system_prompt")
"""

__version__ = "0.1.0"
__author__ = "SimpleAudit Contributors"

from .model_auditor import ModelAuditor
from .results import AuditResults, AuditResult
from .scenarios import get_scenarios, list_scenario_packs
from .providers import (
    LLMProvider,
    AnthropicProvider,
    OpenAIProvider,
    GrokProvider,
    HuggingFaceProvider,
    OllamaProvider,
    CopilotProvider,
    get_provider,
    PROVIDERS,
)

# Backward compatibility: Auditor is now just an alias for ModelAuditor
# Users should migrate to ModelAuditor, using provider="openai" with base_url for HTTP endpoints
Auditor = ModelAuditor

__all__ = [
    "ModelAuditor",
    "Auditor",  # Deprecated alias for backward compatibility
    "AuditResults", 
    "AuditResult",
    "get_scenarios",
    "list_scenario_packs",
    # Provider classes
    "LLMProvider",
    "AnthropicProvider",
    "OpenAIProvider",
    "GrokProvider",
    "HuggingFaceProvider",
    "OllamaProvider",
    "CopilotProvider",
    "get_provider",
    "PROVIDERS",
]

