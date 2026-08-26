## Results & Analysis

The `simpleaudit.results` and `simpleaudit.repeated_results` modules provide the data structures for storing, analyzing, and exporting audit outcomes. This subsystem handles single-run results, aggregation of multiple runs for stability analysis, and cross-judge comparisons.

### Core Data Structures

#### `AuditResult`
A dataclass representing the outcome of a single audit scenario.

| Field | Type | Description |
| :--- | :--- | :--- |
| `scenario_name` | `str` | Unique identifier for the test scenario. |
| `severity` | `str` | Verdict: `critical`, `high`, `medium`, `low`, `pass`, or `ERROR`. |
| `issues_found` | `List[str]` | Specific deviations identified by the judge. |
| `positive_behaviors` | `List[str]` | Aspects the model handled correctly. |
| `summary` | `str` | Brief professional summary in the target language. |
| `recommendations` | `List[str]` | Concrete improvement suggestions. |
| `token_usage` | `int` fields | Input/output tokens for auditor, judge, and target models. |

**Usage:**
```python
from simpleaudit.results import AuditResult

result = AuditResult(
    scenario_name="test_01",
    severity="pass",
    issues_found=[],
    positive_behaviors=["Correct refusal"],
    summary="Model handled request safely.",
    recommendations=[]
)
```

#### `AuditResults`
A collection of `AuditResult` objects with analysis methods.

**Key Properties:**
*   `score`: Safety score (0–100). Excludes `ERROR` results to prevent infrastructure failures from skewing safety metrics.
*   `severity_distribution`: Count of results per severity level.
*   `token_usage`: Aggregated token counts across all scenarios.
*   `passed` / `failed`: Counts of scenarios with `pass` vs. non-`pass` severity.

**Methods:**
*   `summary()`: Prints a formatted console report including severity distribution, top issues, and token usage.
*   `save(filepath)`: Atomically writes results to JSON. Uses a temporary file and rename to prevent corruption if the process is interrupted.
*   `load(filepath)`: Class method to load results from a JSON file.
*   `plot(save_path)`: Generates a matplotlib visualization (pie chart for severity, bar chart for scenario scores). Requires `matplotlib`.

**Example:**
```python
from simpleaudit.results import AuditResults

results = AuditResults([result1, result2])
results.summary()
results.save("audit_output.json")
```

### Stability Analysis

Repeated runs are necessary to distinguish consistent model behavior from stochastic variance. The `simpleaudit.repeated_results` module provides tools for this.

#### `ScenarioStats`
Per-scenario statistics derived from multiple runs.

| Field | Description |
| :--- | :--- |
| `pass_rate` | Fraction of runs where severity was `pass`. |
| `agreement_rate` | Fraction of runs matching the modal (most common) severity. |
| `normalised_entropy` | Shannon entropy of severity distribution (0.0–1.0). 0.0 indicates unanimous verdicts. |
| `ordinal_spread` | Population standard deviation of severity indices (0–4 scale). |

#### `ModelStabilityReport`
Aggregates statistics for a specific model across multiple runs.

**Key Attributes:**
*   `mean_score`, `std_score`: Mean and standard deviation of the run scores.
*   `cv`: Coefficient of variation (std/mean * 100).
*   `per_scenario`: Dictionary mapping scenario names to `ScenarioStats`.

**Fragility Detection:**
The `fragile(threshold=0.6)` method identifies scenarios where the modal verdict share is below the threshold. A scenario is considered "fragile" if the judge's verdict is unstable across runs. This is critical for identifying scenarios where a single-run result should not be trusted.

**Example:**
```python
from simpleaudit.repeated_results import ModelStabilityReport

# Assuming 'report' is a ModelStabilityReport instance
fragile_scenarios = report.fragile(threshold=0.6)
for name, stats in fragile_scenarios.items():
    print(f"Fragile: {name} (Agreement: {stats.agreement_rate:.2f})")
```

### Cross-Judge Analysis

The `simpleaudit.cross_judge` module allows comparing how different judge models affect severity ratings for the same subject model.

#### `CrossJudgeExperiment`
Orchestrates `AuditExperiment` runs across multiple judge models.

**Parameters:**
*   `models`: List of subject model configurations.
*   `judge_models`: List of judge model configurations (must be ≥ 2).
*   `n_repetitions`: Number of runs per judge/subject combination.
*   `save_dir`: Base directory for results. Results are namespaced by judge label.

#### `CrossJudgeResults`
Container for results from `CrossJudgeExperiment`.

**Methods:**
*   `score_summary()`: Returns per-subject, per-judge score statistics (mean, std, cv).
*   `severity_shifts(subject_label)`: Identifies scenarios where different judges assigned different modal severities. This helps detect if a judge's bias is influencing the outcome.

**Example:**
```python
from simpleaudit.cross_judge import CrossJudgeExperiment

exp = CrossJudgeExperiment(
    models=[{"model": "subject-model", "provider": "anthropic"}],
    judge_models=[
        {"model": "judge-a", "provider": "anthropic"},
        {"model": "judge-b", "provider": "openai"}
    ],
    n_repetitions=3
)

# Run asynchronously
results = await exp.run_async(scenarios="my_scenarios")

# Analyze shifts
shifts = results.severity_shifts("subject-model")
for shift in shifts:
    if shift["shifted"]:
        print(f"Shift detected in {shift['scenario']}: {shift['modals']}")
```

### Best Practices

1.  **Atomic Writes**: Always use `AuditResults.save()` rather than manual JSON dumping to ensure data integrity during interruptions.
2.  **Exclude Errors**: When calculating safety scores, ensure `ERROR` results are excluded. `AuditResults.score` handles this automatically.
3.  **Stability Thresholds**: Use `ModelStabilityReport.fragile()` to filter out unstable scenarios before drawing conclusions from single-run data.
4.  **Judge Bias**: Use `CrossJudgeExperiment` when high-stakes decisions depend on the audit results, to verify that the verdict is not an artifact of a specific judge model.