import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


import pandas as pd
from bq_eda_toolkit.bigquery_visualizer import BigQueryVisualizer

class DummyViz(BigQueryVisualizer):
    def __init__(self, df):
        self.full_table_path = 'x'
        self.columns = list(df.columns)
        self.numeric_columns = ['feat']
        self.categorical_columns = []
        self._df = df
        self.auto_show = False
    def _execute_query(self, q, use_cache=True):
        if 'GROUP BY bin_id' in q:
            return pd.DataFrame({'bin_id':[0,1],'avg_target':[15,45],'n':[3,2]})
        return self._df.copy()


def test_partial_dependence():
    df = pd.DataFrame({'feat':[1,2,3,4,5], 'target':[10,20,30,40,50]})
    viz = DummyViz(df)
    tbl, fig = viz.partial_dependence(feature='feat', target='target', bins=2)
    assert not tbl.empty
    assert fig is not None
