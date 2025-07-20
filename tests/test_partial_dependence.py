import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


import pandas as pd
import polars as pl
from bq_eda_toolkit.bigquery_visualizer import BigQueryVisualizer

class DummyViz(BigQueryVisualizer):
    def __init__(self, df):
        self.full_table_path = 'x'
        self.columns = list(df.columns)
        self.numeric_columns = ['feat']
        self.categorical_columns = []
        self._df = df
        self.auto_show = False

    def get_representative_sample(self, columns=None, max_bytes=None, refresh=False):
        if columns:
            return pl.from_pandas(self._df[columns].copy()).lazy()
        return pl.from_pandas(self._df.copy()).lazy()


def test_partial_dependence():
    df = pd.DataFrame({'feat':[1,2,3,4,5], 'target':[10,20,30,40,50]})
    viz = DummyViz(df)
    tbl, fig = viz.partial_dependence(feature='feat', target='target', bins=2)
    assert list(tbl.columns) == ['bin_id', 'avg_target', 'n']
    assert tbl['n'].sum() == len(df)
    assert fig is not None
