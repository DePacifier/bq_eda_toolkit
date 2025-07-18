import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from bigquery_visualizer import BigQueryVisualizer

class DummyViz(BigQueryVisualizer):
    def __init__(self):
        self.full_table_path = 'x'
        self.numeric_columns = ['num']
        self.columns = ['num']
        self.categorical_columns = []
    def _execute_query(self, q, use_cache=True):
        if 'VAR_SAMP(num)' in q:
            return pd.DataFrame({
                'total_rows':[4],'null_count':[0],'non_null_count':[4],
                'mean':[2.5],'std_dev':[1.12],'variance':[1.25],'min':[1],'max':[4],
                'skewness':[0.0],'kurtosis':[1.5],'quartiles':[[1,1.75,2.5,3.25,4]]})
        return pd.DataFrame({'num':[1,2,3,4]})

def test_numeric_summary_has_skew_kurt():
    viz = DummyViz()
    summary = viz.analyze_numeric_column('num')
    assert 'Skewness' in summary
    assert 'Kurtosis' in summary
    df = viz.analyze_all_numeric().data
    assert 'Skewness' in df.columns
