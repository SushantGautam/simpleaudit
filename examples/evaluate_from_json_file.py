"""
Example: Load and evaluate conversations from a JSON file.

This shows how to load your existing conversations from a JSON file
and evaluate them using SimpleAudit.

Run with:
    python examples/evaluate_from_json_file.py
"""

import json
from simpleaudit import evaluate_conversations

# Load conversations from JSON file
with open("examples/sample_conversations.json", "r") as f:
    conversations_data = json.load(f)

print(f"Loaded {len(conversations_data)} conversations from JSON file\n")

# Show what we loaded
for conv in conversations_data:
    print(f"- {conv['scenario_name']}: {len(conv['conversation'])} messages")

print("\nEvaluating with judge...\n")

# Evaluate the conversations
# Note: This requires an API key. Set environment variable:
#   export ANTHROPIC_API_KEY=your-key-here
# Or use provider="openai" with OPENAI_API_KEY
# Or use provider="ollama" with a local model (no API key needed!)

try:
    results = evaluate_conversations(
        conversations_data,
        judge_provider="anthropic",  # Change to "openai", "grok", or "ollama"
        # judge_model="claude-sonnet-4-20250514",  # Optional: specify judge model
        # judge_api_key="sk-...",  # Optional: provide API key directly
        # For local testing without API key, use:
        # judge_provider="ollama",
        # judge_model="llama3.2"
    )
    
    # Display results
    results.summary()
    
    # Save detailed results
    results.save("evaluation_results.json")
    print("\n✓ Saved detailed results to evaluation_results.json")
    
except Exception as e:
    print(f"Error: {e}")
    print("\nTo run this example:")
    print("1. Set ANTHROPIC_API_KEY environment variable, or")
    print("2. Use judge_provider='openai' with OPENAI_API_KEY, or")
    print("3. Use judge_provider='ollama' with a local model (free!)")
    print("\nExample with Ollama (no API key needed):")
    print("  # First: ollama serve && ollama pull llama3.2")
    print("  results = evaluate_conversations(conversations_data, judge_provider='ollama', judge_model='llama3.2')")
