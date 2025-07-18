import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from bigquery_visualizer import BigQueryVisualizer

class DummyViz(BigQueryVisualizer):
    def __init__(self, df):
        self.full_table_path = 'x'
        self.columns = list(df.columns)
        self.numeric_columns = []
        self._df = df
        self.rep_sample_df = df.copy()
    def _execute_query(self, q, use_cache=True):
        return self._df.copy()

    def get_representative_sample(self, columns=None, max_bytes=None, refresh=False):
        df = self.rep_sample_df.copy()
        if columns:
            df = df[columns]
        return df


def test_missingness_functions():
    df = pd.DataFrame({
        'a':[1,None,3,None],
        'b':[None,2,None,4],
        'c':[1,2,None,None],
    })
    viz = DummyViz(df)
    corr = viz.missingness_correlation()
    assert corr.shape == (3,3)

    combos = viz.frequent_missing_patterns(top_n=2)
    assert len(combos) == 2

    mask, fig, mcar = viz.missingness_map()
    assert mask.shape == df.shape
    assert not mcar.empty
