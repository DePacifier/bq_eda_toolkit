import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


import pandas as pd
import polars as pl
from pathlib import Path
from bq_eda_toolkit.bigquery_visualizer import BigQueryVisualizer

class DummyViz(BigQueryVisualizer):
    def __init__(self, cache_dir=Path("/tmp/test_cache"), clear_cache=True):
        self.full_table_path = 'x'
        self._query_cache = {}
        self.max_result_bytes = 100
        self.schema_df = pd.DataFrame({
            'column_name': ['a', 'b'],
            'data_type': ['INT64', 'STRING'],
            'category': ['numeric', 'string'],
        })
        self.columns = ['a', 'b']
        self.numeric_columns = ['a']
        self.categorical_columns = ['b']
        self.string_columns = ['b']
        self.boolean_columns = []
        self.datetime_columns = []
        self.complex_columns = []
        self.geographic_columns = []
        self.client = None
        self.sample_cache_dir = Path(cache_dir)
        self.sample_cache_dir.mkdir(exist_ok=True)
        if clear_cache:
            self.__class__.clear_sample_cache(disk=True, cache_dir=self.sample_cache_dir)
        self._calls = 0
        self.auto_show = False
    def _execute_query(self, q, use_cache=True):
        self.last_query = q
        self._calls += 1
        return pd.DataFrame({'a':[1], 'b':['x']})
    def evaluate_sample_bias(self, sample_rows=1000, alpha=0.05):
        self.bias_called = True
        return pd.DataFrame()

    def get_representative_sample(self, columns=None, max_bytes=None, refresh=False):
        return super().get_representative_sample(columns=columns, max_bytes=max_bytes, refresh=refresh)

def test_representative_sample_cached():
    viz = DummyViz()
    df1 = viz.get_representative_sample()
    assert isinstance(df1, pl.LazyFrame)
    assert 'LIMIT 3' in viz.last_query
    assert viz.bias_called
    calls = viz._calls
    df2 = viz.get_representative_sample()
    assert isinstance(df2, pl.LazyFrame)
    assert viz._calls == calls
    assert df1.collect().to_pandas().equals(df2.collect().to_pandas())

def test_representative_sample_refresh():
    viz = DummyViz()
    viz.get_representative_sample()
    calls = viz._calls
    viz.get_representative_sample(refresh=True)
    assert viz._calls == calls + 1


def test_sample_cache_autoreload(tmp_path):
    viz1 = DummyViz(cache_dir=tmp_path)
    df1 = viz1.get_representative_sample()
    assert isinstance(df1, pl.LazyFrame)
    assert viz1._calls == 1

    DummyViz.clear_sample_cache()

    viz2 = DummyViz(cache_dir=tmp_path, clear_cache=False)
    df2 = viz2.get_representative_sample()
    assert isinstance(df2, pl.LazyFrame)
    assert viz2._calls == 0
    assert df1.collect().to_pandas().equals(df2.collect().to_pandas())


def test_clear_sample_cache(tmp_path):
    viz = DummyViz(cache_dir=tmp_path)
    viz.get_representative_sample()
    assert list(tmp_path.glob("rep_*.parquet"))
    DummyViz.clear_sample_cache(disk=True, cache_dir=tmp_path)
    assert not list(tmp_path.glob("rep_*.parquet"))
