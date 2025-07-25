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
        if 'COVAR_' in q:
            return pd.DataFrame({'c1':['a'], 'c2':['b'], 'cov':[2.5]})
        return pd.DataFrame()

    def get_representative_sample(self, columns=None, max_bytes=None, refresh=False):
        return pd.DataFrame({'a':[1,2], 'b':[3,4]})

def test_numeric_covariances():
    viz = DummyViz()
    mat = viz.numeric_covariances(['a','b'])
    assert mat.loc['a','b'] == 2.5
    assert mat.loc['b','a'] == 2.5
