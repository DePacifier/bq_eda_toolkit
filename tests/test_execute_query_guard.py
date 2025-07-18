import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import pytest

from bigquery_visualizer import BigQueryVisualizer, QueryExecutionError

class FakePlanEntry:
    def __init__(self, est):
        self._properties = {"statistics": {"estimatedBytes": est}}

class FakeDryJob:
    def __init__(self, scanned=0):
        self.total_bytes_processed = scanned
    def result(self):
        pass

class FakePlanJob:
    def __init__(self, est):
        self.query_plan = [FakePlanEntry(est)]
    def result(self):
        pass

class FakeResultJob:
    def __init__(self, df):
        self._df = df
    def to_dataframe(self):
        return self._df

class DummyClient:
    def __init__(self, est_bytes, df, raise_error=False):
        self.est_bytes = est_bytes
        self.df = df
        self.raise_error = raise_error
    def query(self, q, job_config=None):
        if job_config is not None:
            return FakeDryJob()
        if q.startswith("EXPLAIN"):
            return FakePlanJob(self.est_bytes)
        if self.raise_error:
            raise RuntimeError("boom")
        return FakeResultJob(self.df)

class DummyViz(BigQueryVisualizer):
    def __init__(self, est_bytes=0, df=None, cache_threshold=100, raise_error=False):
        self.project_id = 'p'
        self.table_id = 'd.t'
        self.full_table_path = 'x'
        self._query_cache = {}
        self.max_bytes_scanned = 10_000
        self.max_result_bytes = 2_000_000_000
        self.cache_threshold_bytes = cache_threshold
        self.client = DummyClient(est_bytes, df if df is not None else pd.DataFrame(), raise_error=raise_error)
        self.auto_show = False


def test_result_size_guard_raises():
    viz = DummyViz(est_bytes=3_000_000_000, df=pd.DataFrame())
    with pytest.raises(RuntimeError):
        viz._execute_query("SELECT 1")


def test_big_df_not_cached():
    df = pd.DataFrame({'a': range(1000)})
    viz = DummyViz(est_bytes=0, df=df, cache_threshold=1)
    out = viz._execute_query("SELECT 1")
    assert out.equals(df)
    assert "SELECT 1" not in viz._query_cache


def test_query_errors_raise():
    viz = DummyViz(est_bytes=0, df=pd.DataFrame(), raise_error=True)
    with pytest.raises(QueryExecutionError):
        viz._execute_query("SELECT 1")
