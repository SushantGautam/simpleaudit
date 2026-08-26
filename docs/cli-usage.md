## CLI Usage

The `simpleaudit` command-line interface (CLI) provides tools for visualizing and exporting AI safety audit results. It serves as the primary interface for developers to interact with generated audit data without writing custom Python scripts. The CLI is implemented in `simpleaudit/cli.py` and exposes two main subcommands: `serve` and `export-html` [1].

### Overview

The CLI module defines the `main()` function as the entry point. It uses `argparse` to parse command-line arguments and routes execution to specific handlers based on the selected subcommand [1]. The tool is designed for post-audit analysis, allowing users to inspect results via a local web server or generate static HTML reports for sharing.

Both subcommands require optional dependencies for visualization. If these are missing, the CLI exits with an error message instructing the user to install them via `pip install 'simpleaudit[visualize]'` [1].

### Command: `serve`

The `serve` command starts a local web server to visualize audit results. This is useful for interactive exploration of multiple audit runs.

**Syntax:**
```bash
simpleaudit serve [OPTIONS]
```

**Arguments:**

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--results_dir` | `str` | `None` | Directory containing JSON result files. If not specified, defaults to the current directory (`.`) with a warning [1]. |
| `--port` | `int` | `8000` | Port number for the web server [1]. |
| `--host` | `str` | `127.0.0.1` | Host address to bind the server to [1]. |

**Behavior:**
1.  The CLI attempts to import `start_server` from `simpleaudit.visualization.server`.
2.  If `--results_dir` is omitted, the CLI prints a warning recommending explicit specification to avoid confusion [1].
3.  The server starts and serves the JSON files found in the specified directory.

**Example:**
```bash
# Serve results from a specific directory on port 5000
simpleaudit serve --results_dir ./audit_results --port 5000

# Serve results from current directory on default port
simpleaudit serve
```

### Command: `export-html`

The `export-html` command creates a standalone HTML file with audit results inlined. This file can be opened directly in any web browser without a server or network connection, making it ideal for sharing reports or archiving results.

**Syntax:**
```bash
simpleaudit export-html <json_path> [OPTIONS]
```

**Arguments:**

| Argument/Flag | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `json_path` | `str` | Yes | Path to the audit results JSON file [1]. |
| `-o`, `--output` | `str` | No | Output HTML file path. If omitted, defaults to the input JSON path with the extension changed to `.html` [1]. |

**Behavior:**
1.  The CLI attempts to import `export_standalone_html` from `simpleaudit.visualization.server`.
2.  If `--output` is not specified, the output path is derived from the input `json_path` by replacing the file extension with `.html` [1].
3.  The CLI validates the existence of the input JSON file. If the file is not found, it exits with an error [1].
4.  Upon success, it prints the path to the generated HTML file and a note that it can be opened directly in a browser [1].

**Error Handling:**
*   **FileNotFoundError:** Exits with code 1 and prints "Error: {json_path} does not exist" [1].
*   **ValueError:** Exits with code 1 and prints the specific error message [1].
*   **ModuleNotFoundError:** Exits with code 1 and prompts to install visualization dependencies [1].

**Example:**
```bash
# Export results to a default named HTML file
simpleaudit export-html results/audit_run_1.json

# Export results to a specific output file
simpleaudit export-html results/audit_run_1.json -o report.html
```

### Dependencies and Installation

The CLI visualization features rely on optional dependencies. If these are not installed, both `serve` and `export-html` will fail with a `ModuleNotFoundError`.

To install the required dependencies:
```bash
pip install 'simpleaudit[visualize]'
```

### Configuration and Environment

The CLI itself does not directly read API keys, as it operates on pre-generated JSON results. However, the underlying audit tools that generate these results rely on environment variables for API configuration.

Common environment variables used by the auditing engine include:
*   `OPENAI_API_KEY`: For OpenAI models (default provider)
*   `ANTHROPIC_API_KEY`: For Anthropic models
*   `SIMPLEAUDIT_VISUALIZER_SECRET`: Optional secret for the visualizer, read from environment [3]

Refer to the [any-llm-sdk documentation](https://mozilla-ai.github.io/any-llm/providers) for the complete list of supported providers and their specific environment variables [6].

### Privacy and Security

SimpleAudit is a local developer tool. The CLI and visualization components:
*   Do not collect, store, or transmit personal data [2].
*   Do not phone home to external servers [2].
*   Process only local files (JSON results) and synthetic test data [2].

Users should ensure that the JSON result files they visualize do not contain sensitive information if they are shared via the exported HTML files. The tool does not automatically redact content; users are responsible for managing the confidentiality of their audit data [2].

### Troubleshooting

**Error: `ModuleNotFoundError`**
*   **Cause:** Missing visualization dependencies.
*   **Solution:** Run `pip install 'simpleaudit[visualize]'` [1].

**Error: `FileNotFoundError`**
*   **Cause:** The specified `json_path` does not exist.
*   **Solution:** Verify the file path and ensure the audit run completed successfully [1].

**Warning: `--results_dir not specified`**
*   **Cause:** Using `simpleaudit serve` without specifying a results directory.
*   **Solution:** Explicitly pass `--results_dir <path>` to avoid scanning the current directory unintentionally [1].

### Reference

*   **Source File:** `simpleaudit/cli.py` [1]
*   **Visualization Module:** `simpleaudit/visualization/server.py` (imported dynamically) [1]
*   **Documentation:** See `/README.md` for full API reference and `/examples/` for Jupyter notebooks [2].