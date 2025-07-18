import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from pathlib import Path
from bigquery_visualizer import BigQueryVisualizer

class DummyViz(BigQueryVisualizer):
    def __init__(self):
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
        self.sample_cache_dir = Path("/tmp/test_cache")
        self.sample_cache_dir.mkdir(exist_ok=True)
        self.rep_sample_columns_key = None
        self.rep_sample_df = None
        self._calls = 0
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
    assert 'LIMIT 3' in viz.last_query
    assert viz.bias_called
    calls = viz._calls
    df2 = viz.get_representative_sample()
    assert viz._calls == calls
    assert df1.equals(df2)

def test_representative_sample_refresh():
    viz = DummyViz()
    viz.get_representative_sample()
    calls = viz._calls
    viz.get_representative_sample(refresh=True)
    assert viz._calls == calls + 1
