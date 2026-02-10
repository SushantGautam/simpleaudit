# Evaluating Existing Conversations

## Quick Answer: YES, it's supported! 

You can use SimpleAudit's judge to evaluate your existing conversations JSON.

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

## That's it!

The judge will evaluate each conversation and provide:
- **Severity levels**: critical, high, medium, low, or pass
- **Issues found**: List of specific problems
- **Recommendations**: Suggestions for improvement
- **Safety score**: Overall score out of 100

## Choose Your Judge

```python
# Use OpenAI as judge (default: Claude)
results = evaluate_conversations(
    conversations_data,
    judge_provider="openai",
    judge_model="gpt-4o"
)

# Use Grok as judge
results = evaluate_conversations(
    conversations_data,
    judge_provider="grok",
    judge_model="grok-3"
)

# Use local Ollama as judge (free!)
results = evaluate_conversations(
    conversations_data,
    judge_provider="ollama",
    judge_model="llama3.2"
)

# Provide API key directly
results = evaluate_conversations(
    conversations_data,
    judge_provider="openai",
    judge_api_key="sk-..."
)
```

## Optional: Add Scenario Descriptions

If you want more context for the judge:

```python
conversations_data = [
    {
        "scenario_name": "HELPMed_S1_Meningitis_Incomplete_Info",
        "scenario_description": "Test if the system provides safe medical advice",  # Add this
        "conversation": [...]
    }
]
```

## Complete Example

See `examples/evaluate_existing_conversations.py` for a full working example with sample conversations.
