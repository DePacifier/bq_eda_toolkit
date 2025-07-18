import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


import pandas as pd
from bq_eda_toolkit.bigquery_visualizer import BigQueryVisualizer

class DummyViz(BigQueryVisualizer):
    def __init__(self):
        self.full_table_path = 'x'
        self.columns = ['a', 'b']
        self.numeric_columns = ['a', 'b']
        self.categorical_columns = []
        self.auto_show = False

    def _execute_query(self, q, use_cache=True):
        if 'CORR(' in q:
            return pd.DataFrame({'c1':['a'], 'c2':['b'], 'corr':[0.7]})
        if 'ML.PCA' in q:
            return pd.DataFrame({'dim1':[0.1], 'dim2':[0.2]})
        if 'APPROX_QUANTILES' in q:
            return pd.DataFrame({'bucket':[1], 'bin_start':[0], 'bin_end':[5], 'n':[10]})
        return pd.DataFrame()

    def get_representative_sample(self, columns=None, max_bytes=None, refresh=False):
        return pd.DataFrame({'a':[1,2], 'b':[3,4]})

def test_numeric_correlations():
    viz = DummyViz()
    mat = viz.numeric_correlations(['a','b'])
    assert mat.loc['a','b'] == 0.7
    assert mat.loc['b','a'] == 0.7

def test_project_2d_sql():
    viz = DummyViz()
    df, fig = viz.project_2d(method='pca', columns=['a','b'])
    assert not df.empty
    assert 'dim1' in df.columns and 'dim2' in df.columns
