# SimpleAudit Developer Documentation

SimpleAudit framework audits LLM outputs. Judges evaluate scenarios. Results visualize audit findings.

## Table of Contents

- [Getting Started](./getting-started.md) — Installation, environment setup, first audit run.
- [Core Architecture](./core-architecture.md) — High-level system design, data flow, module interactions.
- [CLI Usage](./cli-usage.md) — Command-line interface commands, flags, execution modes.
- [Judges System](./judges-system.md) — Evaluation logic, scoring mechanisms, judge implementations.
- [Custom Judges](./custom-judges.md) — Extending framework with new evaluation criteria, logic.
- [Scenarios Overview](./scenarios-overview.md) — Built-in test scenarios, domain coverage, scenario structure.
- [Creating Custom Scenarios](./creating-custom-scenarios.md) — Guidelines, patterns for building new adversarial test scenarios.
- [BullshitBench Integration](./bullshitbench-integration.md) — Implementation details for BullshitBench health, version scenarios.
- [Results & Analysis](./results-analysis.md) — Data structures for audit results, aggregation, repeated runs.
- [Judge Validation](./judge-validation.md) — Methodology for validating judge reliability, comparative scoring.
- [Visualization Server](./visualization-server.md) — Local web interface for inspecting audit results, visualizations.
- [Reframing & Utilities](./reframing-utilities.md) — Text reframing techniques, shared utility functions.
- [API Reference](./api-reference.md) — Complete class, function reference for all public modules.