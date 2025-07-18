import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from bigquery_visualizer import BigQueryVisualizer

class DummyViz(BigQueryVisualizer):
    def __init__(self, df):
        self.full_table_path = 'x'
        self.columns = list(df.columns)
        self.numeric_columns = ['num']
        self.categorical_columns = []
        self._df = df
    def _execute_query(self, q, use_cache=True):
        return self._df.copy()

def test_histogram_with_kde():
    df = pd.DataFrame({'num':[1,2,2,3,4,5]})
    viz = DummyViz(df)
    _, fig = viz.plot_histogram(numeric_column='num', kde=True, bins=5)
    assert fig is not None
    assert len(fig.data) > 1
