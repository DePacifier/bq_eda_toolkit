# BigQuery EDA Toolkit

> **Turn a raw BigQuery table into a model-ready insight deck in minutes.**

* Plotly-powered charts  
* SQL-first, Python-light: heavy lifting stays in BigQuery  
* Modular pipeline (run everything) **and** standalone helpers  
* Built-in query cache + cost guard (no surprise bills)

---

## ✨ Key features

| module | highlight |
|--------|-----------|
| `bigquery_visualizer.py` | 1-line connection ➜ 20+ plotting / analysis helpers (bar, scatter, histogram, violin, pie, sunburst, descriptive stats …) |
| `stages/` | Each `Stage` runs one slice of EDA (profiling, quality, univariate, …) and writes artefacts into a shared `AnalysisContext`. |
| `pipeline.py` | Orchestrator that executes any list of stages: `Pipeline().run(viz)`. |
| `analysis_context.py` | In-memory store for result tables & figures—later export to HTML / Markdown. |
| cost guard | Dry-run every query; abort if bytes scanned > configurable limit (default 1 GB). |
| cache | In-memory DataFrame cache per notebook session—re-plots are instant, no extra BigQuery cost. |
| `feature_advice.py` | Auto-suggest encoding, scaling and interaction plans based on profiling results. |

---

## 📦 Installation

```bash
git clone https://github.com/DePacifier/bq_eda_toolkit.git
cd bq_eda_toolkit
pip install -r requirements.txt
```

## 🚀 Usage

```python
from bigquery_visualizer import BigQueryVisualizer
from pipeline import Pipeline

viz = BigQueryVisualizer(
    project_id="my-project",
    table_id="dataset.table",
    credentials_path="path/to/key.json",
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
```

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
