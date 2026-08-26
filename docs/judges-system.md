## Judges System

SimpleAudit judges evaluate AI model responses using LLM-as-a-judge methodologies. Each judge config defines a `probe_prompt` (generates test inputs) and `judge_prompt` (evaluates outputs). Judges emit structured JSON judgments containing scores, severity levels, or categorical flags.

### Architecture

The `simpleaudit.judges` module exports `JUDGE_CONFIGS`, a dictionary mapping string names to judge configuration objects. Each config contains:

- `name`: Human-readable identifier
- `description`: One-line summary
- `probe_prompt`: Template for generating adversarial or test user messages
- `judge_prompt`: Instructions for the evaluation LLM
- `output_schema`: Documentation of expected JSON fields
- `response_schema`: Machine-readable JSON Schema for strict validation
- `source`: Academic references and methodology notes

### Public API

#### `get_judge(name: str) -> Dict[str, Any]`

Retrieves a built-in judge configuration by name. Returns a shallow copy to prevent accidental mutation of the shared registry.

```python
from simpleaudit.judges import get_judge

config = get_judge("safety")
print(config["judge_prompt"][:100])
```

**Raises:** `ValueError` if `name` is not in `JUDGE_CONFIGS`.

#### `list_judge_configs() -> Dict[str, str]`

Returns a mapping of available judge names to their descriptions.

```python
from simpleaudit.judges import list_judge_configs

for name, desc in list_judge_configs().items():
    print(f"{name}: {desc}")
```

### Available Judges

| Judge Name | Methodology | Output Type | Key Fields |
| :--- | :--- | :--- | :--- |
| `safety` | Constitutional AI (Bai et al., 2022) | Severity | `severity`, `issues_found`, `summary` |
| `abstention` | AbstentionBench (Kirichenko et al., 2025) | Score (1–10) | `score`, `abstained`, `appropriate`, `category` |
| `helpfulness` | MT-Bench (Zheng et al., 2023) | Score (1–10) | `score`, `relevance`, `accuracy`, `clarity`, `completeness` |
| `factuality` | G-Eval (Liu et al., 2023) | Score (1–10) | `score`, `verdict`, `factual_errors` |
| `harm` | HELM Safety (Liang et al., 2022) | Categorical | `harmful`, `category`, `severity` |
| `binary_abstention` | Custom Binary Classifier | Boolean | `abstained`, `reasoning` |
| `judge_conviction` | Meta-Judge Observer | Verdict Extraction | `verdict`, `changed`, `reasoning` |

### Scoring and Severity Mapping

Score-based judges (`abstention`, `helpfulness`, `factuality`) emit integer scores 1–10. The framework derives `AuditResult.severity` automatically:

- **9–10**: `pass`
- **7–8**: `low`
- **5–6**: `medium`
- **3–4**: `high`
- **1–2**: `critical`

This mapping ensures consistency across summaries and visualizations. The raw judgment dictionary remains unchanged in results.

### Usage Examples

#### Basic Integration

Pass a judge name to `ModelAuditor`:

```python
from simpleaudit import ModelAuditor

auditor = ModelAuditor(
    model="gpt-4o",
    judge="helpfulness",
    probe_prompt="Ask about Python best practices."
)

results = auditor.run(scenarios)
for r in results:
    print(r.judgment["score"], r.judgment["relevance"])
```

#### Custom Probe with Named Judge

Override the default probe prompt while keeping the judge logic:

```python
auditor = ModelAuditor(
    model="claude-3-sonnet",
    judge="factuality",
    probe_prompt="Ask about historical dates. Include one false premise."
)
```

#### Fully Custom Judge

Provide a custom `judge_prompt` to bypass built-in configurations:

```python
auditor = ModelAuditor(
    model="gpt-4o",
    judge_prompt="You are a strict code reviewer. Check for PEP8 violations. Output JSON: {\"score\": int, \"issues\": list}"
)
```

#### Binary Abstention Detection

Use `binary_abstention` for deterministic refusal detection:

```python
auditor = ModelAuditor(
    model="llama-3-70b",
    judge="binary_abstention"
)

# Post-process to check appropriateness against ground truth
for result in auditor.run(scenarios):
    if result.judgment["abstained"] != scenario.metadata["expected_outcome"] == "refuse":
        print("Mismatch:", scenario.id)
```

### Judge Config Structure

Each judge config is a dictionary with the following keys:

```python
{
    "name": str,
    "description": str,
    "probe_prompt": str | None,  # None for meta-judges
    "judge_prompt": str,
    "output_schema": Dict[str, str],
    "response_schema": Dict[str, Any],  # JSON Schema
    "source": Dict[str, Any],
    "metadata": Dict[str, Any]
}
```

#### `response_schema` Importance

The `response_schema` defines the strict JSON structure the judge must return. When `json_format=True` is set in `ModelAuditor`, the framework enforces this schema. Without it, the judge may fail to emit required fields.

Example (`factuality`):

```json
{
  "type": "object",
  "properties": {
    "score": {"type": "integer"},
    "verdict": {"type": "string", "enum": ["accurate", "partially_accurate", "inaccurate"]},
    "factual_errors": {"type": "array", "items": {"type": "string"}},
    "unsupported_claims": {"type": "array", "items": {"type": "string"}},
    "reasoning": {"type": "string"}
  },
  "required": ["score", "verdict", "factual_errors", "unsupported_claims", "reasoning"]
}
```

### Methodology References

- **Safety**: Constitutional AI principles (harm avoidance, boundaries, accuracy, transparency, manipulation resistance).
- **Abstention**: AbstentionBench taxonomy (answer_unknown, false_premise, stale, subjective, underspecified_context, underspecified_intent).
- **Helpfulness**: MT-Bench dimensions (relevance, accuracy, clarity, completeness).
- **Factuality**: G-Eval chain-of-thought evaluation for hallucination detection.
- **Harm**: HELM Safety six categories (violence, fraud, discrimination, sexual, harassment, deception).

### Meta-Judges

`judge_conviction` is a meta-judge used by `WiggleRunner` to track verdict stability under pressure. It extracts the current verdict from a candidate judge's free-form response and reports whether it changed.

```python
from simpleaudit.judges import get_judge

config = get_judge("judge_conviction")
# config["judge_prompt"] is the observer system prompt
# config["response_schema"] defines {verdict, changed, reasoning}
```

This judge does not evaluate correctness; it only extracts the stated verdict for post-processing analysis.