# Summary: Evaluate Existing Conversations Feature

## ✅ Implementation Complete

### What Was Added

A new `evaluate_conversations()` function that allows users to evaluate pre-existing conversations using a judge LLM, without needing to generate new probes or run against a target system.

### Answer to User's Question

**Q: "I already have conversations JSON for different scenarios. Can I use a judge and evaluate them? Is it already supported in this project?"**

**A: YES! It's now fully supported!**

## Minimal Code Example

```python
from simpleaudit import evaluate_conversations

# Your conversations data (exactly in your format)
conversations_data = [
    {
        "scenario_name": "HELPMed_S1_Meningitis_Incomplete_Info",
        "conversation": [
            {
                "role": "user",
                "content": "Hey, I'm at the m. . . "
            },
            {
                "role": "assistant",
                "content": ". . . ."
            }
            # ... more messages
        ]
    }
    # ... more scenarios
]

# Evaluate with judge
results = evaluate_conversations(conversations_data)

# View results
results.summary()
results.save("results.json")
```

## Function Signature

```python
def evaluate_conversations(
    conversations_data: List[Dict],
    judge_provider: str = "anthropic",
    judge_api_key: Optional[str] = None,
    judge_model: Optional[str] = None,
    judge_base_url: Optional[str] = None,
    verbose: bool = True,
    prompt_for_key: bool = True,
) -> AuditResults
```

### Parameters (All with `judge_*` Prefix for Clarity)

- **conversations_data**: List of conversation dicts with `scenario_name` and `conversation`
- **judge_provider**: LLM provider ("anthropic", "openai", "grok", "ollama", etc.)
- **judge_api_key**: Optional API key (or use environment variables)
- **judge_model**: Optional model name (defaults vary by provider)
- **judge_base_url**: Optional custom base URL
- **verbose**: Show progress during evaluation
- **prompt_for_key**: Prompt for API key if not found

## What You Get

The judge evaluates each conversation and provides:

- ✅ **Severity levels**: critical, high, medium, low, or pass
- ✅ **Issues found**: List of specific problems
- ✅ **Positive behaviors**: What the model did well
- ✅ **Recommendations**: Suggestions for improvement
- ✅ **Safety score**: Overall score out of 100
- ✅ **Full audit reports**: Save to JSON, generate plots

## Files Added/Modified

### New Files
1. **simpleaudit/auditor.py** - Added `evaluate_conversations()` function
2. **examples/evaluate_existing_conversations.py** - Full working example
3. **examples/evaluate_from_json_file.py** - Loading from JSON file
4. **examples/sample_conversations.json** - Sample data format
5. **tests/test_evaluate_conversations.py** - Comprehensive tests (5 tests, all passing)
6. **docs/EVALUATE_EXISTING_CONVERSATIONS.md** - Detailed guide

### Updated Files
1. **simpleaudit/__init__.py** - Exported `evaluate_conversations`
2. **README.md** - Added "Evaluating Existing Conversations" section

## Testing

✅ All 5 new tests passing:
- test_evaluate_conversations_basic
- test_evaluate_conversations_with_issues
- test_evaluate_conversations_missing_scenario_description
- test_evaluate_conversations_provider_options
- test_evaluate_conversations_results_summary

✅ No security vulnerabilities found (CodeQL scan clean)
✅ Code review feedback addressed

## Usage Examples

### Basic Usage
```python
results = evaluate_conversations(conversations_data)
```

### With OpenAI as Judge
```python
results = evaluate_conversations(
    conversations_data,
    judge_provider="openai",
    judge_model="gpt-4o"
)
```

### With Local Ollama (Free!)
```python
results = evaluate_conversations(
    conversations_data,
    judge_provider="ollama",
    judge_model="llama3.2"
)
```

### Load from JSON File
```python
import json
with open("conversations.json") as f:
    conversations = json.load(f)
results = evaluate_conversations(conversations)
```

## Documentation

- **Quick Guide**: `docs/EVALUATE_EXISTING_CONVERSATIONS.md`
- **Full Example**: `examples/evaluate_existing_conversations.py`
- **JSON File Example**: `examples/evaluate_from_json_file.py`
- **Sample Data**: `examples/sample_conversations.json`
- **README Section**: "Evaluating Existing Conversations"

## Key Design Decisions

1. **Clear Parameter Names**: All judge-related parameters use `judge_*` prefix for clarity
2. **Optional scenario_description**: If not provided, a default one is generated
3. **Same Output Format**: Returns `AuditResults` object like other audit methods
4. **Consistent API**: Matches the pattern of `Auditor` and `ModelAuditor` classes
5. **Progress Feedback**: Uses tqdm progress bars for visibility

## Next Steps for Users

1. Import the function: `from simpleaudit import evaluate_conversations`
2. Format your conversations as shown above
3. Call the function with your data
4. Review results with `results.summary()`
5. Save results with `results.save("file.json")`
6. Plot results with `results.plot()`
