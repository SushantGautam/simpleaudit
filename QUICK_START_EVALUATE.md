# Quick Start: Evaluate Your Existing Conversations

## Your Question
> "I already have conversations JSON for different scenarios. Can I use a judge and evaluate them?"

## Answer: YES! ✅

## Minimal Code (Copy & Paste Ready)

```python
from simpleaudit import evaluate_conversations

# Your conversations (exactly your format!)
conversations = [
    {
        "scenario_name": "HELPMed_S1_Meningitis_Incomplete_Info",
        "conversation": [
            {"role": "user", "content": "Hey, I'm at the m..."},
            {"role": "assistant", "content": "...."}
        ]
    },
    # ... more scenarios
]

# Evaluate
results = evaluate_conversations(conversations)

# View results
results.summary()
```

That's it! 🎉

## From JSON File

```python
import json
from simpleaudit import evaluate_conversations

with open("your_conversations.json") as f:
    conversations = json.load(f)

results = evaluate_conversations(conversations)
results.summary()
results.save("evaluation_results.json")
```

## Choose Your Judge

```python
# OpenAI (if you prefer GPT over Claude)
results = evaluate_conversations(
    conversations,
    judge_provider="openai"
)

# Local Ollama (completely free!)
results = evaluate_conversations(
    conversations,
    judge_provider="ollama",
    judge_model="llama3.2"
)
```

## What You Get

- 🔍 Severity: critical / high / medium / low / pass
- 📋 Issues found
- ✅ Positive behaviors  
- 💡 Recommendations
- 📊 Safety score (0-100)

## See Full Examples

- `examples/evaluate_existing_conversations.py` - Complete working example
- `examples/evaluate_from_json_file.py` - Loading from file
- `docs/EVALUATE_EXISTING_CONVERSATIONS.md` - Full documentation

## Need Help?

Set your API key:
```bash
export ANTHROPIC_API_KEY=your-key-here
# or
export OPENAI_API_KEY=your-key-here
```

Or use local Ollama (no API key needed):
```bash
ollama serve
ollama pull llama3.2
```
