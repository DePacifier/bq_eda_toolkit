import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


import pandas as pd
import polars as pl
from bq_eda_toolkit.bigquery_visualizer import BigQueryVisualizer

class DummyViz(BigQueryVisualizer):
    def __init__(self):
        self.full_table_path = 'x'
        self.columns = ['num']
        self.numeric_columns = ['num']
        self.categorical_columns = []
        self.auto_show = False

    def _execute_query(self, q, use_cache=True):
        if 'APPROX_QUANTILES' in q:
            return pd.DataFrame({
                'bucket':[1,2],
                'bin_start':[0,5],
                'bin_end':[5,10],
                'n':[2,4]
            })
        return pd.DataFrame()

    def get_representative_sample(self, columns=None, max_bytes=None, refresh=False):
        return pl.DataFrame({'num':[1,2,2,3,4,5]}).lazy()

def test_histogram_with_kde():
    viz = DummyViz()
    _, fig = viz.plot_histogram(numeric_column='num', kde=True, bins=5)
    assert fig is not None
    assert len(fig.data) > 1
