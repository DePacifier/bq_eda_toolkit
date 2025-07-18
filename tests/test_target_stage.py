import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


import pandas as pd
from bq_eda_toolkit.analysis_context import AnalysisContext
from bq_eda_toolkit.stages.core_stages import TargetStage

class DummyViz:
    def __init__(self):
        self.full_table_path = 'x'
        self.columns = ['label','feat']
        self.numeric_columns = ['feat']
        self.categorical_columns = ['label']
        self.plot_histogram = lambda **k: (pd.DataFrame(), None)
        self.plot_pie_chart = lambda dimension: (pd.DataFrame(), None)
    def _execute_query(self, q, use_cache=True):
        if 'GROUP BY label' in q:
            return pd.DataFrame({'label':['A','B'], 'n':[80,20]})
        if 'WHERE label IS NOT NULL' in q:
            return pd.DataFrame({'label':['A','B','A'], 'feat':[1,2,3]})
        return pd.DataFrame()


def test_target_stage_class_balance():
    ctx = AnalysisContext(params={'target_column':'label'})
    viz = DummyViz()
    TargetStage().run(viz, ctx)
    balance = ctx.get_table('target.class_balance')
    assert not balance.empty
    assert balance['n'].sum() == 100
