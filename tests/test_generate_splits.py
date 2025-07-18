import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


import pandas as pd
from bq_eda_toolkit.bigquery_visualizer import BigQueryVisualizer

class DummyViz(BigQueryVisualizer):
    def __init__(self):
        self.project_id = 'p'
        self.table_id = 'ds.tbl'
        self.full_table_path = 'p.ds.tbl'
        self._query_cache = {}
        self.auto_show = False
        self.queries = []
    def _execute_query(self, q, use_cache=True):
        self.queries.append(q)
        if q.lstrip().startswith('SELECT'):
            return pd.DataFrame({
                'split': ['train', 'validation', 'test'],
                'cls': ['A', 'A', 'A'],
                'n': [8, 1, 1]
            })
        return pd.DataFrame()

def test_generate_splits_sql_and_balance():
    viz = DummyViz()
    splits, balance = viz.generate_splits(target_column='label')
    assert splits == {
        'train': 'ds.tbl_train',
        'validation': 'ds.tbl_val',
        'test': 'ds.tbl_test'
    }
    assert 'CREATE OR REPLACE TABLE ds.tbl_train' in viz.queries[0]
    assert 'CREATE OR REPLACE TABLE ds.tbl_val' in viz.queries[1]
    assert 'CREATE OR REPLACE TABLE ds.tbl_test' in viz.queries[2]
    assert viz.queries[-1].lstrip().startswith('SELECT')
    assert not balance.empty
    assert 'train' in balance.columns and 'validation' in balance.columns
