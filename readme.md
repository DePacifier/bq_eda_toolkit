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

---

## 📦 Installation

```bash
git clone https://github.com/DePacifier/bq_eda_toolkit.git
cd bq_eda_toolkit
pip install -r requirements.txt
