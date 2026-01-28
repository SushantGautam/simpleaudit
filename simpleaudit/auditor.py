"""
Core Auditor class for SimpleAudit.

This module contains the main Auditor class that orchestrates the audit process
using an LLM as both the adversarial probe generator and safety judge.

This class is now a convenience wrapper around ModelAuditor for auditing
OpenAI-compatible HTTP endpoints.

Supports multiple LLM providers:
- Anthropic (Claude) - default
- OpenAI (GPT-4, GPT-5, etc.)
- Grok (xAI)
- HuggingFace (local transformers)
- Ollama (local models)
"""

from typing import List, Dict, Optional, Union

from .model_auditor import ModelAuditor
from .results import AuditResults
from .client import TargetClient


class Auditor:
    """
    Main auditor class that uses an LLM to probe and evaluate AI systems.
    
    This class is a convenience wrapper around ModelAuditor for auditing
    OpenAI-compatible HTTP endpoints. It uses OpenAIProvider with a custom
    base_url to connect to the target endpoint.
    
    Args:
        target: URL of the target API endpoint (OpenAI-compatible chat completions)
        provider: LLM provider to use for probe generation and judging: 
                 "anthropic" (default), "openai", or "grok"
        api_key: API key for the chosen provider (or use env vars:
                 ANTHROPIC_API_KEY, OPENAI_API_KEY, or XAI_API_KEY)
        model: Model to use for probe generation and judging
        base_url: Custom base URL for the probe generator LLM (optional)
        judge_provider: LLM provider for evaluation (defaults to same as provider)
        judge_api_key: API key for judge provider (defaults to api_key if same provider)
        judge_model: Model to use for judge (defaults to model if not specified)
        judge_base_url: Custom base URL for judge provider (optional)
        target_model: Model name to send to target API (default: "default")
        target_api_key: API key for the target model endpoint (optional)
        max_turns: Maximum conversation turns per scenario (default: 5)
        timeout: Request timeout in seconds (default: 120)
        verbose: Print progress during audit (default: True)
        prompt_for_key: If True, prompt for API key if not found (default: True)
        
        # Legacy parameters (for backwards compatibility)
        anthropic_api_key: Deprecated, use api_key instead
    
    Example:
        >>> # Using Anthropic (default)
        >>> auditor = Auditor("http://localhost:8000/v1/chat/completions")
        >>> results = auditor.run("safety")
        
        >>> # Using OpenAI for probe generation
        >>> auditor = Auditor(
        ...     "http://localhost:8000/v1/chat/completions",
        ...     provider="openai"
        ... )
        
        >>> # Using custom OpenAI-compatible endpoint
        >>> auditor = Auditor(
        ...     "http://localhost:8000/v1/chat/completions",
        ...     provider="openai",
        ...     base_url="http://localhost:8000/v1",
        ...     judge_provider="openai",
        ...     judge_base_url="http://localhost:8000/v1"
        ... )
    """
    
    def __init__(
        self,
        target: str,
        provider: str = "anthropic",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        judge_provider: Optional[str] = None,
        judge_api_key: Optional[str] = None,
        judge_model: Optional[str] = None,
        judge_base_url: Optional[str] = None,
        target_model: str = "default",
        target_api_key: Optional[str] = None,
        max_turns: int = 5,
        timeout: float = 120.0,
        verbose: bool = True,
        prompt_for_key: bool = True,
        # Legacy parameter for backwards compatibility
        anthropic_api_key: Optional[str] = None,
    ):
        self.target_url = target
        self.max_turns = max_turns
        self.verbose = verbose
        
        # Handle legacy anthropic_api_key parameter
        if anthropic_api_key is not None:
            api_key = anthropic_api_key
            if provider == "anthropic":
                pass  # Already using anthropic
        
        # Initialize target client for health check compatibility
        self.target = TargetClient(target, model=target_model, timeout=timeout, api_key=target_api_key)
        
        # Create ModelAuditor with OpenAI provider for the target endpoint
        # The target endpoint is accessed via OpenAI provider with custom base_url
        # Extract base_url from target if not provided
        target_base_url = target
        if "/chat/completions" in target:
            target_base_url = target.split("/chat/completions")[0]
        
        # Create the ModelAuditor instance
        # - Use "openai" provider with target's base_url for the target model
        # - Use specified provider for probe generation and judging
        self._model_auditor = ModelAuditor(
            provider="openai",  # Use OpenAI provider for target (OpenAI-compatible endpoint)
            base_url=target_base_url,
            model=target_model,
            api_key=target_api_key or "dummy-key",  # OpenAI provider requires an API key
            system_prompt="",  # No system prompt for HTTP endpoint target
            judge_provider=judge_provider or provider,  # Probe generator and judge
            judge_api_key=judge_api_key or api_key,
            judge_model=judge_model or model,
            judge_base_url=judge_base_url or base_url,
            max_turns=max_turns,
            verbose=verbose,
            prompt_for_key=prompt_for_key,
        )
        
        # Store references for backward compatibility
        self.provider = self._model_auditor.judge_provider  # Probe generator
        self.model = self._model_auditor.judge_model
        self.judge_provider = self._model_auditor.judge_provider  # Same as probe generator
        self.judge_model = self._model_auditor.judge_model
    
    def run_scenario(
        self, 
        name: str, 
        description: str, 
        max_turns: Optional[int] = None,
        language: str = "English",
        pbar_audit = None,
        pbar_judge = None,
    ):
        """
        Run a single audit scenario.
        
        Delegates to the internal ModelAuditor instance.
        """
        return self._model_auditor.run_scenario(
            name=name,
            description=description,
            max_turns=max_turns,
            language=language,
            pbar_audit=pbar_audit,
            pbar_judge=pbar_judge,
        )
    
    def run(
        self,
        scenarios: Union[str, List[Dict]],
        max_turns: Optional[int] = None,
        language: str = "English",
        max_workers: int = 1
    ) -> AuditResults:
        """
        Run audit with given scenarios.
        
        Delegates to the internal ModelAuditor instance.
        
        Args:
            scenarios: Either a scenario pack name ("safety", "rag", "health") 
                      or a list of custom scenario dicts with 'name' and 'description'
            max_turns: Override default max_turns for all scenarios
            language: Language for probe generation (default: English)
            max_workers: Number of parallel workers for scenarios (default: 1)
        
        Returns:
            AuditResults object with all results and analysis methods
        
        Example:
            # Use built-in scenarios
            results = auditor.run("safety")
            
            # Use custom scenarios
            results = auditor.run([
                {"name": "My Test", "description": "Test if the system..."}
            ], max_workers=4)
        """
        return self._model_auditor.run(
            scenarios=scenarios,
            max_turns=max_turns,
            language=language,
            max_workers=max_workers,
        )
    
    def check_connection(self) -> bool:
        """Test connection to target API."""
        return self.target.health_check()
    
    def _log(self, message: str, name: Optional[str] = None):
        """Print message if verbose mode is enabled (thread-safe and tqdm-friendly)."""
        self._model_auditor._log(message, name)
