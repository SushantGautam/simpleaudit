## Reframing & Utilities

The `simpleaudit.reframing` and `simpleaudit.utils` modules provide mechanisms for validating audit robustness and standardizing data handling. The reframing module isolates judge prompt variability from target model variability, while the utilities module offers shared functions for severity normalization, JSON parsing, and media handling.

### Reframing Module

The `reframing` module addresses a specific artifact in AI auditing: prompt sensitivity. Standard reproducibility checks resample conversations, varying both the target model's output and the judge's grading. This conflates target instability with judge instability. The reframing check holds the transcript fixed and varies only the judge prompt wording. If a verdict flips between semantically equivalent judge prompts, the measurement is an apparatus artifact, not a finding about the target model.

This process costs only judge tokens, as transcripts are already stored. It does not call the target model.

#### Key Classes

**`PromptVariant`**
Represents a single wording of a judge prompt. Variants must be supplied explicitly; the system does not generate paraphrases to avoid introducing uncontrolled variables into the measurement instrument.

*   `label` (str): Unique identifier for the variant.
*   `judge_prompt` (str): The full text of the judge prompt.
*   `response_schema` (Optional[Dict[str, Any]]): Optional schema for structured output.

**`StoredRecord`**
Contains the transcript and context required for re-grading.

*   `scenario_name` (str): Identifier for the test scenario.
*   `scenario_description` (str): Contextual description for the judge.
*   `conversation` (List[Dict[str, Any]]): The chat history to be graded.
*   `expected_behavior` (Optional[List[str]]): Criteria for expected model behavior.

**`ReframingResults`**
Aggregates verdicts across all scenario-variant pairs.

*   `variant_labels` (List[str]): Order of variants executed.
*   `per_scenario` (Dict[str, Dict[str, str]]): Maps scenario names to variant labels and their resulting severities.
*   `judgments` (Dict[str, Dict[str, Dict[str, Any]]]): Raw judge outputs for traceability.
*   `input_tokens`, `output_tokens` (int): Token usage for the judge model.

Methods:
*   `shifts()`: Returns a list of dictionaries detailing per-scenario verdicts. Includes `shifted` (bool) indicating if variants disagreed, and `direction` (int) indicating severity difference (positive means second variant is stricter).
*   `invariant_rate()`: Returns the fraction of scenarios where all variants produced identical verdicts.

#### Functions

**`load_stored_records(source)`**
Loads transcripts from a saved audit result file or dictionary. Skips entries with empty conversations.

*   `source` (Union[str, Path, Dict[str, Any]]): Path to JSON file or parsed payload.
*   Returns: `List[StoredRecord]`

**`reframing_check(...)`**
Synchronous wrapper for re-grading transcripts. Cannot be called from an active event loop.

**`reframing_check_async(...)`**
Asynchronous implementation of the reframing check.

Parameters:
*   `judge_client`: AnyLLM client instance.
*   `judge_model`: Model identifier for the judge.
*   `records`: Sequence of `StoredRecord` objects.
*   `variants`: Sequence of `PromptVariant` objects (minimum 2).
*   `json_format` (bool): Whether to request JSON output (default: True).
*   `max_retries` (int): Retry count for judge calls.
*   `retry_backoff` (float): Delay between retries.

#### Example Usage

```python
from any_llm import AnyLLM
from simpleaudit.judges import get_judge
from simpleaudit.reframing import load_stored_records, reframing_check, PromptVariant

# Retrieve base judge prompt
base_prompt = get_judge("safety")["judge_prompt"]

# Define variants (must be explicit, not model-generated)
variants = [
    PromptVariant("baseline", base_prompt),
    PromptVariant("reordered", reordered_rubric_text),
]

# Load existing audit results
records = load_stored_records("results/my_audit.json")

# Execute check
results = reframing_check(
    judge_client=AnyLLM.create("anthropic"),
    judge_model="claude-opus-4-7",
    records=records,
    variants=variants
)

# Report shifts
for entry in results.shifts():
    if entry["shifted"]:
        print(f"{entry['scenario']}: {entry['modals']} -> {entry['direction']}")
```

### Utilities Module

The `utils` module contains shared functions for data normalization and parsing used across the framework.

#### Severity Management

**Constants**
*   `VALID_SEVERITIES`: Set of canonical severity levels: `{"critical", "high", "medium", "low", "pass"}`.
*   `SEVERITY_ORDER`: Ordered list from least to most severe: `["pass", "low", "medium", "high", "critical"]`.

**`normalize_severity(severity)`**
Maps judge-emitted severity strings to the canonical ladder. Handles case sensitivity, whitespace, and aliases (e.g., "none" maps to "pass"). Returns "medium" for `None` inputs.

**`severity_direction(severity_a, severity_b)`**
Calculates the positional difference between two severities on `SEVERITY_ORDER`. Returns a positive integer if `severity_b` is stricter, negative if less strict, and `None` if either value is outside the canonical ladder.

**`severity_from_score(score, max_score=10.0)`**
Maps numeric scores (1-10) to severity levels.
*   9-10: pass
*   7-8: low
*   5-6: medium
*   3-4: high
*   1-2: critical

Returns `None` if the score is not numeric.

#### JSON Parsing

**`parse_json_response(response, default_severity="ERROR")`**
Robustly parses JSON from LLM responses. Handles markdown code blocks, leading/trailing text, and malformed JSON.

Returns a dictionary with guaranteed keys:
*   `severity`: Normalized severity level.
*   `issues_found`: List of identified issues.
*   `positive_behaviors`: List of positive actions.
*   `summary`: Text summary of the response.
*   `recommendations`: List of suggestions.

If parsing fails completely, it attempts best-effort extraction of severity indicators from natural language before returning default values.

#### Media Handling

**`image_media_type(file_uri)`**
Resolves a URI to an image media type. Raises `ValueError` if the file is not recognized as an image.

**`image_data_uri(file_uri)`**
Reads an image and converts it to a base64 data URI. Uses `fsspec` to support local paths, HTTP(S), S3, and GCS. Results are cached using `lru_cache` to avoid re-reading files during multi-turn audits.

**`image_content_block(file_uri)`**
Constructs an OpenAI-style content block for image attachments.

```python
block = image_content_block("s3://bucket/images/example.png")
# Returns: {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
```

### Best Practices

1.  **Explicit Variants**: Always define `PromptVariant` instances manually. Do not use LLMs to generate paraphrases for reframing checks, as this introduces uncontrolled variance.
2.  **Severity Normalization**: Always pass raw judge outputs through `normalize_severity` before storing or comparing results to ensure consistency across different judge models.
3.  **Cache Management**: The `image_data_uri` function caches results. If image files change between audit runs, clear the cache or restart the process to ensure fresh data is loaded.
4.  **Error Handling**: `parse_json_response` is designed to fail gracefully. However, if `severity` is frequently defaulting to "ERROR", review the judge prompt to ensure it outputs valid JSON.