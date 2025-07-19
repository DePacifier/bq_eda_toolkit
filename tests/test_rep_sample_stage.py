import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pandas as pd
from bq_eda_toolkit.stages.core_stages import RepSampleStage
from bq_eda_toolkit.bigquery_visualizer import BigQueryVisualizer

class DummyViz(BigQueryVisualizer):
    def __init__(self):
        self.full_table_path = 'ds.tbl'
        self.table_rows = 1000
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
        self.auto_show = False

    def _execute_query(self, q, use_cache=True):
        self.last_query = q
        return pd.DataFrame({'a':[1],'b':['x']})


def test_tablesample_query_generation():
    viz = DummyViz()
    stage = RepSampleStage(n=100, columns=['a','b'])
    q = stage.build_query(viz)
    assert 'TABLESAMPLE SYSTEM (10 PERCENT)' in q
    assert 'LIMIT 100' in q


def test_stratified_query_generation():
    viz = DummyViz()
    stage = RepSampleStage(n=100, method='STRATIFIED', stratify_by='b', columns=['a','b'])
    q = stage.build_query(viz)
    assert 'ROW_NUMBER() OVER(PARTITION BY b' in q
    assert 'CEIL(cnt * 10 / 100)' in q


def test_auto_limit_from_max_bytes():
    viz = DummyViz()
    stage = RepSampleStage(columns=['a','b'])
    q = stage.build_query(viz)
    assert 'LIMIT 3' in q


def test_explicit_sample_percent():
    viz = DummyViz()
    stage = RepSampleStage(sample_percent=5, columns=['a','b'])
    q = stage.build_query(viz)
    assert 'TABLESAMPLE SYSTEM (5 PERCENT)' in q
