## API Reference

SimpleAudit is a lightweight framework for red-teaming AI systems using LLMs as both auditor and judge. It supports direct API auditing via providers like OpenAI, Anthropic, Grok, Ollama, and vLLM. The library provides tools for scenario management, judge configuration, result analysis, and stability testing.

### Core Classes

#### `ModelAuditor`
Primary class for auditing a specific LLM model via its API.

**Parameters:**
*   `model` (str): Target model name (e.g., "gpt-4o-mini").
*   `provider` (str): Target model provider (e.g., "openai", "anthropic").
*   `judge_model` (str): Judge model name.
*   `judge_provider` (str): Judge model provider.
*   `api_key` (Optional[str]): API key for target model.
*   `base_url` (Optional[str]): Base URL for target model (for local/custom endpoints).
*   `system_prompt` (Optional[str]): System prompt for the target model.
*   `judge` (Optional[str]): Name of a built-in judge config (e.g., "safety", "helpfulness").
*   `judge_prompt` (Optional[str]): Custom judge prompt. Overrides `judge` config if provided.
*   `probe_prompt` (Optional[str]): Custom probe prompt. Overrides `judge` config if provided.
*   `judge_response_schema` (Optional[Dict]): Custom JSON schema for judge output.
*   `judge_fields` (Optional[List[str]]): Restrict judge output to specific fields (e.g., ["severity", "summary"]).
*   `json_format` (bool): Force JSON output format (default: True).
*   `max_turns` (int): Maximum conversation turns (default: 5).
*   `verbose` (bool): Enable verbose logging.
*   `show_progress` (bool): Show progress bars (default: True).
*   `max_retries` (int): Max retry attempts for API calls (default: 2).
*   `retry_backoff` (float): Backoff time between retries (default: 0.5).

**Example:**
```python
from simpleaudit import ModelAuditor, get_scenarios

auditor = ModelAuditor(
    model="gpt-4o-mini",
    provider="openai",
    judge_model="gpt-4o",
    judge_provider="openai",
    judge="safety",
    system_prompt="You are a helpful assistant."
)
results = auditor.run(get_scenarios("safety"))
results.summary()
```

#### `AuditExperiment`
Runs audits across multiple models and supports repeated runs for stability analysis.

**Parameters:**
*   `models` (List[Dict]): List of model configurations. Each dict must contain `model` and optionally `label`, `provider`, `api_key`, etc.
*   `judge_model` (Optional[str]): Default judge model for all targets.
*   `judge_provider` (Optional[str]): Default judge provider.
*   `judge` (Optional[str]): Default judge config name.
*   `n_repetitions` (int): Number of times to run each scenario (default: 1).
*   `adaptive_reruns` (Optional[Dict]): Config for adaptive reruns based on agreement.
*   `save_dir` (Optional[str]): Directory to save results for resuming.
*   `on_model_done` (Optional[Callable]): Callback invoked after each model completes.

**Example:**
```python
from simpleaudit import AuditExperiment, get_scenarios

experiment = AuditExperiment(
    models=[
        {"model": "gpt-4o-mini", "label": "GPT-4o Mini", "provider": "openai"},
        {"model": "claude-3-haiku", "label": "Claude Haiku", "provider": "anthropic"}
    ],
    judge_model="gpt-4o",
    judge_provider="openai",
    judge="safety",
    n_repetitions=3,
    save_dir="./audit_results"
)
results = experiment.run(get_scenarios("safety"))
```

### Result Classes

#### `AuditResults`
Collection of `AuditResult` objects with analysis methods.

**Properties:**
*   `score`: Safety score (0-100) excluding ERROR results.
*   `severity_distribution`: Dict mapping severity levels to counts.
*   `token_usage`: Dict of token counts for auditor, judge, and target.
*   `all_issues`: Deduplicated list of all issues found.
*   `all_recommendations`: Deduplicated list of all recommendations.

**Methods:**
*   `summary()`: Prints a formatted summary to stdout.
*   `save(filepath)`: Saves results to a JSON file atomically.
*   `load(filepath)`: Class method to load results from a JSON file.
*   `plot(save_path)`: Generates a matplotlib visualization of results.

#### `AuditResult`
Dataclass representing the outcome of a single scenario audit.

**Fields:**
*   `scenario_name`: Name of the scenario.
*   `severity`: One of "critical", "high", "medium", "low", "pass", "ERROR".
*   `issues_found`: List of specific issues.
*   `positive_behaviors`: List of positive behaviors observed.
*   `summary`: Short professional summary.
*   `recommendations`: List of improvement suggestions.
*   `conversation`: List of message dicts representing the chat history.
*   `judgment`: Raw JSON dict from the judge.

### Utility Functions

#### `get_scenarios(pack_name: str) -> List[Dict]`
Retrieves a list of scenario dictionaries from a built-in pack.

**Available Packs:**
*   `"safety"`: General AI safety scenarios.
*   `"rag"`: RAG-specific scenarios.
*   `"health"`: Healthcare domain scenarios.
*   `"bullshitbench"`: Combined BullshitBench v1+v2 scenarios.
*   `"all"`: All standard scenarios combined.

**Example:**
```python
scenarios = get_scenarios("safety")
```

#### `list_scenario_packs() -> Dict[str, int]`
Returns a dictionary mapping pack names to the number of scenarios in each.

#### `get_judge(name: str) -> Dict[str, Any]`
Retrieves a built-in judge configuration by name.

**Available Judges:**
*   `"safety"`: Constitutional AI safety evaluation.
*   `"helpfulness"`: MT-Bench style quality evaluation.
*   `"factuality"`: Hallucination detection.
*   `"abstention"`: Refusal appropriateness.

**Example:**
```python
judge_config = get_judge("safety")
```

#### `list_judge_configs() -> Dict[str, str]`
Returns a dictionary mapping judge names to their descriptions.

### Stability and Cross-Judge Analysis

#### `RepeatedExperimentResults`
Aggregates results from multiple repetitions of an experiment. Provides stability metrics and variance analysis.

#### `CrossJudgeExperiment`
Compares the outputs of different judge models on the same scenarios to assess judge reliability.

#### `reframing_check`
Tests if a model's behavior changes when a scenario is reframed or paraphrased.

### Configuration Notes

*   **Provider Defaults:** If `provider` is not specified, it defaults to `"openai"`.
*   **Judge Precedence:** Explicit `judge_prompt` and `probe_prompt` parameters override the `judge` config. `judge_fields` overrides the response schema.
*   **Atomic Writes:** Result files are written atomically to prevent corruption during interrupts.
*   **Error Handling:** API errors result in `severity="ERROR"`, which is excluded from the `score` calculation but included in `severity_distribution`.

### Installation

```bash
pip install simpleaudit
```

### Dependencies

*   `any_llm`: For multi-provider LLM access.
*   `tqdm`: For progress bars.
*   `matplotlib`: For plotting (optional).