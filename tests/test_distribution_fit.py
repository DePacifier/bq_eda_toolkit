import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pandas as pd
from bq_eda_toolkit.bigquery_visualizer import BigQueryVisualizer

class DummyViz(BigQueryVisualizer):
    def __init__(self):
        self.full_table_path = 'x'
        self.columns = ['num']
        self.numeric_columns = ['num']
        self.categorical_columns = []
        self.auto_show = False
    def _execute_query(self, q, use_cache=True):
        if 'SELECT num' in q:
            return pd.DataFrame({'num': [1,2,3,4,5,6,7,8,9,10]})
        return pd.DataFrame()

def test_fit_distribution():
    viz = DummyViz()
    summary, model = viz.fit_distribution(numeric_column='num')
    assert not summary.empty
    assert 'name' in summary.columns
    assert model is not None
