import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pandas as pd
from bq_eda_toolkit.analysis_context import AnalysisContext
from bq_eda_toolkit.stages.comparison import DatasetComparisonStage
from bq_eda_toolkit.bigquery_visualizer import BigQueryVisualizer

class DummyViz(BigQueryVisualizer):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.full_table_path = 'x'
        self.columns = list(df.columns)
        self.numeric_columns = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        self.categorical_columns = [c for c in df.columns if c not in self.numeric_columns]
        self.string_columns = self.categorical_columns.copy()
        self.boolean_columns = []
        self.datetime_columns = []
        self.complex_columns = []
        self.geographic_columns = []
        self.auto_show = False

    def fetch_sample(self, n, *, where=None):
        return self.df.sample(min(n, len(self.df)), random_state=42)


def test_dataset_comparison_stage_detects_drift():
    df1 = pd.DataFrame({
        'num': list(range(100)) + list(range(100)),
        'cat': ['A'] * 150 + ['B'] * 50,
    })
    df2 = pd.DataFrame({
        'num': list(range(50, 150)) + list(range(50, 150)),
        'cat': ['A'] * 40 + ['B'] * 160,
    })

    viz1 = DummyViz(df1)
    viz2 = DummyViz(df2)

    ctx = AnalysisContext()
    DatasetComparisonStage(viz2, sample_rows=50).run(viz1, ctx)

    drift = ctx.get_table('comparison.drift_tests')
    assert drift is not None
    assert not drift.empty
    assert any(drift['drift'])
