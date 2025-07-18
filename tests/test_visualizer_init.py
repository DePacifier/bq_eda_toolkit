import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


import pandas as pd
from bq_eda_toolkit.bigquery_visualizer import BigQueryVisualizer
from google.cloud import bigquery

class DummyClient:
    def __init__(self, *args, **kwargs):
        pass


def test_init_sets_table_info(monkeypatch):
    monkeypatch.setattr(bigquery, "Client", lambda *a, **k: DummyClient())

    class DummyViz(BigQueryVisualizer):
        def _execute_query(self, q, use_cache=True):
            if "INFORMATION_SCHEMA.COLUMNS" in q:
                return pd.DataFrame({
                    "column_name": ["a"],
                    "data_type": ["INT64"],
                    "category": ["numeric"],
                })
            if "INFORMATION_SCHEMA.TABLES" in q:
                return pd.DataFrame({"row_count": [10], "size_bytes": [1_000_000_000]})
            return pd.DataFrame()

    viz = DummyViz(project_id="p", table_id="ds.tbl")
    assert viz.table_rows == 10
    assert abs(viz.table_size_gb - 1.0) < 1e-6
