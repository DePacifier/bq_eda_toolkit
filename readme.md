# BigQuery EDA Toolkit

> **Turn a raw BigQuery table into a model-ready insight deck in minutes.**

* Plotly-powered charts  
* SQL-first, Python-light: heavy lifting stays in BigQuery  
* Modular pipeline (run everything) **and** standalone helpers  
* Built-in query cache + cost guard (dry-run + result-size check)

---

## ✨ Key features

| module | highlight |
|--------|-----------|
| `bigquery_visualizer.py` | 1-line connection ➜ 20+ plotting / analysis helpers (bar, scatter, histogram, violin, pie, sunburst, descriptive stats, sample-based partial dependence …) |
| `stages/` | Each `Stage` runs one slice of EDA (profiling, quality, univariate, …) and writes artefacts into a shared `AnalysisContext`. |
| `pipeline.py` | Orchestrator that executes any list of stages: `Pipeline().run(viz)`. |
| `analysis_context.py` | In-memory store for result tables & figures—later export to HTML / Markdown. |
| cost guard | Dry-run every query; abort if bytes scanned > configurable limit (default 1 GB) and if `EXPLAIN` estimates the result exceeds `max_result_bytes` (default 2 GB). |
| cache | DataFrames are cached per session only when their memory usage is below `cache_threshold_bytes` (default 100 MB). |
| `feature_advice.py` | Auto-suggest encoding, scaling and interaction plans and trains a quick BQML linear regression for diagnostics. |
| `RepSampleStage` | Build representative samples via `TABLESAMPLE` or stratified sampling. |

---

## 📦 Installation

```bash
pip install bq-eda-toolkit
# or for the latest development version
pip install git+https://github.com/DePacifier/bq_eda_toolkit.git
```

## 🚀 Usage

```python
import logging
from bq_eda_toolkit.bigquery_visualizer import BigQueryVisualizer
from bq_eda_toolkit.pipeline import Pipeline

logging.basicConfig(level=logging.INFO)

viz = BigQueryVisualizer(
    project_id="my-project",
    table_id="dataset.table",
    credentials_path="path/to/key.json",
    # optional guards:
    # max_result_bytes=2_000_000_000,
    # cache_threshold_bytes=100_000_000,
)

# prints: e.g. "ℹ️ 100000 rows · 2.5 GB"
print(f"{viz.table_rows} rows, {viz.table_size_gb:.2f} GB")

# run the default EDA pipeline
Pipeline().run(viz)

# check if a small sample is representative
bias = viz.evaluate_sample_bias(sample_rows=1000)
print(bias)

# create stratified train/validation/test splits
splits, balance = viz.generate_splits(
    target_column="label",
    method="stratified",
)
print(balance)

# run a custom query with guard logic only
from bq_eda_toolkit.utils.bq_executor import execute_query_with_guard
row_count_df = execute_query_with_guard(
    viz.client,
    "SELECT COUNT(*) AS n FROM dataset.table",
)

# Use DatasetComparisonStage to check for distribution drift between tables
# results are stored under "comparison.drift_tests"
other = BigQueryVisualizer(project_id="my-project", table_id="dataset.previous")
ctx = Pipeline(stages=[DatasetComparisonStage(other)]).run(viz)
print(ctx.get_table("comparison.drift_tests"))
```

Visualisation functions return Plotly or Matplotlib objects. They do not call
``.show()`` themselves, so display or save the returned figure when needed.

### Accessing results

Each stage stores its output tables and figures in an `AnalysisContext` instance.
For example:

```python
ctx = Pipeline().run(viz)
ctx.get_table("quality.unique_ratio")          # unique-value ratios
ctx.get_table("quality.categorical_quality")   # categorical singleton stats
ctx.get_table("target.class_balance")          # distribution of target classes
ctx.get_table("feature_advice.encoding_plan")  # suggested encodings
ctx.get_table("feature_advice.imputation_plan")  # how to fill missing values
```

The `FeatureAdviceStage` consumes earlier statistics to auto‑generate
encoding, imputation and scaling suggestions, plus a list of possible
interaction terms. These tables can guide feature engineering for a
machine learning model.
Running this stage requires the BigQuery ML API to be enabled.

### Extending the pipeline

The toolkit is modular: each `Stage` encapsulates one slice of the analysis and
`Pipeline` simply runs them in sequence. You can plug in your own logic by
subclassing `BaseStage` and adding the stage to the pipeline. Ad‑hoc SQL is
supported via `execute_query_with_guard`, which powers `BigQueryVisualizer`'s
internal query method and enforces the dry‑run and result‑size checks.

```python
from bq_eda_toolkit.stages.base import BaseStage
from bq_eda_toolkit.utils.bq_executor import execute_query_with_guard

class MyStage(BaseStage):
    def run(self, viz, ctx):
        df = execute_query_with_guard(
            viz.client,
            "SELECT COUNT(*) AS n FROM dataset.table",
        )
        ctx.add_table(self.key("row_count"), df)

pipe = Pipeline(stages=[MyStage("custom")])
ctx = pipe.run(viz)
```

Results are shared via the `AnalysisContext`, so custom stages can consume or
produce tables just like the built‑in ones.

## 🛠️ Configuration

BigQuery credentials can be supplied either via the `credentials_path` argument or by setting the `GOOGLE_APPLICATION_CREDENTIALS` environment variable. The visualizer also exposes query guard parameters:

- `max_bytes_scanned`: abort if a dry run would scan more than this many bytes (default `10_000_000_000`).
- `max_result_bytes`: abort if `EXPLAIN` estimates the result exceeds this size (default `2_000_000_000`).
- `cache_threshold_bytes`: only cache DataFrames below this threshold (default `100_000_000`).

You may set these as environment variables in your workflow or pass them directly when instantiating `BigQueryVisualizer`.

## 🔐 Credential management

Store your service-account JSON key outside the repository. Point `credentials_path` or `GOOGLE_APPLICATION_CREDENTIALS` to the file via an environment variable or a secrets manager. To rotate credentials, create a new key in the GCP console, update the stored secret or variable to reference the new file and revoke the old key.

## 🧪 Tests & contributing

Install dependencies and run the test suite with `pytest`:

```bash
pip install -r requirements.txt
pip install pytest
pytest
```

Please ensure the tests pass before submitting a pull request. Contributions are welcome via GitHub PRs.

