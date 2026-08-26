## Core Architecture

SimpleAudit is a Python library for auditing Large Language Models (LLMs) using a "LLM-as-a-Judge" methodology. It evaluates target models against specific scenario packs, using separate judge models to assess responses for safety, accuracy, and integrity. The system is designed for reproducibility, supporting direct API calls to various providers (OpenAI, Anthropic, Ollama, vLLM) without requiring an external server.

### System Design

The architecture follows a three-tier interaction model:

1.  **Target Model**: The LLM being audited. It receives prompts from scenario packs and generates responses.
2.  **Judge Model**: An LLM that evaluates the target model's responses. It outputs structured verdicts (severity, issues, recommendations) based on a defined rubric.
3.  **Auditor Model**: An optional LLM that can perform secondary checks or act as a persuader in advanced pressure-testing scenarios. If not specified, it defaults to the Judge Model configuration.

All interactions are mediated by the `ModelAuditor` class, which manages client connections via the `any-llm` library. The `AuditExperiment` class orchestrates multi-model runs, handling caching, retries, and result aggregation.

### Key Modules

#### `simpleaudit.model_auditor`
The core engine for single-model auditing.

**Class: `ModelAuditor`**
Initializes the audit pipeline for a specific target model.

*   **Parameters**:
    *   `model` (str): Model ID for the target.
    *   `provider` (str): Provider for the target (e.g., "openai", "anthropic").
    *   `judge_model` (str): Model ID for the judge.
    *   `judge_provider` (str): Provider for the judge.
    *   `judge` (str, optional): Named judge configuration (e.g., "factuality"). Loads predefined prompts and schemas.
    *   `judge_fields` (List[str], optional): Restrict judge output to specific fields (e.g., `["severity", "summary"]`).
    *   `api_key`, `base_url`: Credentials for the target model.
    *   `judge_api_key`, `judge_base_url`: Credentials for the judge model.
    *   `auditor_model`, `auditor_provider`: Optional separate auditor configuration.
    *   `max_retries` (int): Retry attempts for API failures.
    *   `verbose` (bool): Enable progress logging.

**Key Methods**:
*   `run_async(scenarios, max_turns, language)`: Executes the audit asynchronously. Returns `AuditResults`.
*   `strip_thinking(text)`: Removes reasoning blocks (`<thinking>`) from model outputs.

**Configuration Logic**:
If a named `judge` is provided, `ModelAuditor` loads its default `probe_prompt`, `judge_prompt`, and `response_schema`. Explicit parameters (e.g., `judge_prompt`) override these defaults. If `judge_fields` is set, the response schema is dynamically rebuilt to include only those fields, ensuring strict output formatting.

#### `simpleaudit.experiment`
Orchestrates audits across multiple models and repetitions.

**Class: `AuditExperiment`**
Manages a batch of model audits, handling caching and result persistence.

*   **Parameters**:
    *   `models` (List[Dict]): List of model configurations. Each dict must contain `model` and optionally `label`, `provider`, `api_key`.
    *   `judge_model`, `judge_provider`: Default judge settings applied to all models unless overridden.
    *   `n_repetitions` (int): Number of times to run each scenario for statistical stability.
    *   `adaptive_reruns` (Dict, optional): Configuration for adaptive re-running based on agreement targets.
    *   `save_dir` (str, optional): Directory to cache results for resumable runs.
    *   `on_model_done` (Callable, optional): Callback invoked when a model's audit completes.

**Key Methods**:
*   `run_async(scenarios, max_turns, language, max_workers)`: Runs the experiment. Supports concurrent execution via `max_workers`.
*   `_load_cached_runs(label)`: Retrieves previous results from disk if the configuration fingerprint matches.
*   `_config_fingerprint(...)`: Generates a SHA-256 hash of the current configuration to detect changes.

**Caching Strategy**:
Results are cached per model label in `save_dir/<sanitized_label>/run_<index>.json`. A `config.json` stores the configuration fingerprint. If the fingerprint changes (e.g., different judge model), cached runs are ignored and re-executed. This ensures data integrity when switching judges or prompts.

#### `simpleaudit.utils`
Shared utility functions for data processing and normalization.

**Key Functions**:
*   `normalize_severity(severity)`: Maps judge outputs to the canonical ladder: `pass`, `low`, `medium`, `high`, `critical`. Handles aliases (e.g., "none" -> "pass").
*   `severity_from_score(score, max_score)`: Converts numeric scores (1-10) to severity levels.
    *   9-10: `pass`
    *   7-8: `low`
    *   5-6: `medium`
    *   3-4: `high`
    *   1-2: `critical`
*   `parse_json_response(response)`: Robustly extracts JSON from LLM outputs, handling markdown code blocks and malformed text.
*   `image_data_uri(file_uri)`: Converts image URIs (local, HTTP, S3) to base64 data URIs for multimodal audits.

### Data Flow

1.  **Initialization**: `AuditExperiment` is instantiated with a list of models and judge settings.
2.  **Scenario Selection**: Scenarios are loaded from a pack (e.g., "bullshitbench-v2") or provided as a list of dicts.
3.  **Execution**:
    *   For each model, `ModelAuditor` is initialized.
    *   For each scenario, the target model is prompted.
    *   The judge model evaluates the response using the defined schema.
    *   Results are parsed and normalized.
4.  **Aggregation**: Results are aggregated into `RepeatedExperimentResults`, allowing for statistical analysis across repetitions.
5.  **Persistence**: If `save_dir` is set, results are written to disk for future resumption.

### Example Usage

```python
from simpleaudit import AuditExperiment, ModelAuditor

# Define models to audit
models = [
    {
        "model": "gpt-4o",
        "provider": "openai",
        "label": "GPT-4o"
    },
    {
        "model": "claude-3-sonnet",
        "provider": "anthropic",
        "label": "Claude-3-Sonnet"
    }
]

# Initialize experiment
experiment = AuditExperiment(
    models=models,
    judge_model="gpt-4o",
    judge_provider="openai",
    judge="safety",  # Use predefined safety judge
    n_repetitions=3,
    save_dir="./audit_results"
)

# Run audit
results = experiment.run_async(
    scenarios="bullshitbench-v2",
    max_turns=1,
    language="English"
)

# Access results
for model_label, model_results in results.runs_by_model.items():
    print(f"{model_label}: {model_results[0].summary}")
```

### Configuration Notes

*   **Provider Defaults**: If `provider` is `None`, it defaults to `"openai"`.
*   **Judge Override**: Per-model judge settings in the `models` list override experiment-level defaults.
*   **Header Injection**: `ModelAuditor` does not support custom HTTP headers. Use a reverse proxy (e.g., nginx) to inject required headers if your infrastructure demands them [9].
*   **Citation**: For research use, cite the methodology paper: Gautam et al., "When No Benchmark Exists: Validating Comparative LLM Safety Scoring Without Ground-Truth Labels" (2026) [5].