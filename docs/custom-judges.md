## Custom Judges

SimpleAudit uses **Judges** to evaluate model responses against specific criteria. Judges define the evaluation logic, scoring rubrics, and output schemas. The framework provides built-in judges for common tasks (safety, helpfulness, factuality) and supports fully custom judges via prompt injection or configuration dictionaries.

### Architecture

A Judge configuration is a dictionary containing:
- `name`: Human-readable identifier.
- `description`: One-line summary.
- `probe_prompt`: Instructions for the "prober" model (generates test inputs).
- `judge_prompt`: Instructions for the "judge" model (evaluates responses).
- `output_schema`: Defines expected JSON structure for judge output.
- `source`: Metadata about the judge's origin (e.g., academic paper, domain expert).
- `metadata`: Version, author, language, domain.

Built-in judges are registered in `simpleaudit/judges/__init__.py` under `JUDGE_CONFIGS`.

### Available Built-in Judges

| Name | Description | Output Type |
| :--- | :--- | :--- |
| `safety` | Constitutional AI safety evaluation. | Severity: `critical` \| `high` \| `medium` \| `low` \| `pass` |
| `abstention` | Refusal/abstention appropriateness (AbstentionBench). | Score 1–10, `abstained` flag |
| `helpfulness` | Response quality (MT-Bench dimensions). | Score 1–10, sub-scores |
| `factuality` | Hallucination/factual error detection. | Score 1–10, verdict, error lists |
| `harm` | HELM Safety harm categorisation. | `harmful` flag, category, severity |
| `helsedir_sexhealth_no` | Norwegian sexual-health judge (generic). | Severity: `critical` \| `high` \| `medium` \| `low` \| `pass` |
| `helsedir_sexhealth_no_rag` | Norwegian sexual-health judge (RAG variant). | Severity: `critical` \| `high` \| `medium` \| `low` \| `pass` |
| `binary_abstention` | Binary classifier: did model abstain? | `abstained` (bool), `reasoning` |
| `judge_conviction` | Judge conviction evaluation. | Custom |

**Scoring Logic:**
Score-based judges (`abstention`, `helpfulness`, `factuality`) emit 1–10 scores. The framework derives `AuditResult.severity` automatically:
- 9–10: `pass`
- 7–8: `low`
- 5–6: `medium`
- 3–4: `high`
- 1–2: `critical`

### Using Built-in Judges

Use the `judge` parameter in `ModelAuditor` to select a named judge.

```python
from simpleaudit import ModelAuditor

# Use default probe prompt from config
auditor = ModelAuditor(
    model="gpt-4",
    judge="helpfulness"
)

# Override probe prompt while keeping judge logic
auditor = ModelAuditor(
    model="gpt-4",
    judge="factuality",
    probe_prompt="Ask about historical dates..."
)
```

### Creating Custom Judges

#### Method 1: `judge_prompt` Parameter (Quick Customization)

Pass a custom `judge_prompt` string. This overrides any named judge's prompt. Useful for tweaking evaluation criteria without defining a full config.

```python
from simpleaudit import ModelAuditor

custom_prompt = """
You are a strict grammar checker.
Evaluate the response for grammatical errors.
Return JSON: {"severity": "pass|low|medium|high|critical", "errors": []}
"""

auditor = ModelAuditor(
    model="gpt-4",
    judge_prompt=custom_prompt,
    # No 'judge' name needed; judge_prompt takes precedence
)
```

#### Method 2: Custom Judge Configuration Dict (Full Control)

Define a complete judge configuration dictionary. This allows custom `output_schema`, `probe_prompt`, and metadata.

```python
import json
from simpleaudit import ModelAuditor

custom_judge_config = {
    "name": "Custom Tone Judge",
    "description": "Evaluates if response tone is professional.",
    "probe_prompt": "Ask a question that might trigger a casual response.",
    "judge_prompt": """
        You are a tone evaluator.
        Criteria:
        1. Professionalism: Is the tone formal?
        2. Clarity: Is the message clear?
        
        Return JSON:
        {
            "severity": "pass|low|medium|high|critical",
            "tone_score": 1-10,
            "comments": "string"
        }
    """,
    "output_schema": {
        "severity": "str — one of: critical, high, medium, low, pass",
        "tone_score": "int — 1 to 10",
        "comments": "str — brief explanation"
    },
    "source": {
        "type": "custom",
        "notes": "Internal use case."
    },
    "metadata": {
        "author": "your-team",
        "version": "1.0"
    }
}

# Register or use directly if supported by ModelAuditor API
# Note: Standard API uses 'judge' name or 'judge_prompt'.
# For full dict injection, check if ModelAuditor accepts 'judge_config' or similar.
# Based on source, 'judge' expects a string name. 
# If direct dict injection is not exposed, use 'judge_prompt' for prompt-only 
# or contribute to JUDGE_CONFIGS registry.

# Practical approach: Use 'judge_prompt' for logic, and ensure output 
# matches expected framework severity mapping if using score-based logic.
```

*Note: The standard `ModelAuditor` interface primarily accepts `judge` (string name) or `judge_prompt` (string). To use a full custom dictionary with specific `output_schema` handling, you may need to extend the `JUDGE_CONFIGS` registry in `simpleaudit/judges/__init__.py` or use the `judge_prompt` method if the framework infers schema from prompt instructions.*

### Domain-Specific Example: Norwegian Sexual Health

The `helsedir_sexhealth_no` judge demonstrates complex domain-specific logic. It uses a 1–5 Norwegian severity scale mapped silently to the framework's enum.

**Key Features:**
- **Language:** Norwegian (`no`).
- **Criteria:** 6 weighted criteria (Safety, Scope, Sourcing, Limitations, Manipulation Resistance, Nuance Preservation).
- **Scale Mapping:**
  - 5 (Tilfredsstillende) → `pass`
  - 4 (Mindre avvik) → `low`
  - 3 (Merkbart avvik) → `medium`
  - 2 (Betydelig avvik) → `high`
  - 1 (Uakseptabelt avvik) → `critical`

```python
from simpleaudit import ModelAuditor

auditor = ModelAuditor(
    model="gpt-4",
    judge="helsedir_sexhealth_no"
)

# For RAG-based bots, use the RAG variant
auditor_rag = ModelAuditor(
    model="gpt-4",
    judge="helsedir_sexhealth_no_rag"
)
```

### Utility Functions

#### `get_judge(name: str) -> Dict[str, Any]`

Retrieves a built-in judge configuration by name.

- **Args:** `name` (str): Judge config name (e.g., `"safety"`).
- **Returns:** Judge config dict.
- **Raises:** `ValueError` if name not recognized.

```python
from simpleaudit.judges import get_judge

config = get_judge("safety")
print(config["description"])
```

#### `list_judge_configs() -> Dict[str, str]`

Lists available judge configs and their descriptions.

- **Returns:** Dict mapping config names to one-line descriptions.

```python
from simpleaudit.judges import list_judge_configs

configs = list_judge_configs()
for name, desc in configs.items():
    print(f"{name}: {desc}")
```

### Best Practices

1. **Use Built-ins First:** Start with `safety`, `helpfulness`, or `factuality` for general evaluation.
2. **Custom Prompts for Niche Logic:** Use `judge_prompt` for specific criteria (e.g., tone, formatting) without full config overhead.
3. **Domain Judges:** For regulated or domain-specific contexts (e.g., medical, legal), use specialized judges like `helsedir_sexhealth_no` or create custom configs with explicit `output_schema`.
4. **Severity Mapping:** Ensure custom judges output valid severity values (`critical`, `high`, `medium`, `low`, `pass`) or scores (1–10) that map to these values.
5. **Probe Prompt Override:** Use `probe_prompt` to tailor test inputs to specific scenarios while keeping the judge logic consistent.

### File Paths

- **Registry:** `simpleaudit/judges/__init__.py`
- **Safety Judge:** `simpleaudit/judges/safety.py`
- **Helpfulness Judge:** `simpleaudit/judges/helpfulness.py`
- **Factuality Judge:** `simpleaudit/judges/factuality.py`
- **Norwegian Health Judge:** `simpleaudit/judges/helsedir_sexhealth_no.py`
- **Norwegian Health RAG Judge:** `simpleaudit/judges/helsedir_sexhealth_no_rag.py`