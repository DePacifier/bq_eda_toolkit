import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from analysis_context import AnalysisContext
from stages.core_stages import QualityStage

class DummyViz:
    def __init__(self):
        self.full_table_path = 'x'
        self.columns = ['a', 'num1', 'cat1']
        self.numeric_columns = ['num1']
        self.categorical_columns = ['cat1']
    def _execute_query(self, q, use_cache=True):
        if 'TO_JSON_STRING' in q:
            return pd.DataFrame({'total':[100], 'distinct_rows':[95]})
        if 'COUNT(DISTINCT' in q and 'uniq_' in q:
            return pd.DataFrame({'total':[100], 'uniq_a':[3], 'uniq_num1':[90], 'uniq_cat1':[2]})
        if 'APPROX_QUANTILES' in q:
            return pd.DataFrame({'q1':[0],'q3':[1],'n':[100],'n_out':[5]})
        if 'GROUP BY cat1' in q:
            return pd.DataFrame({'cat1':['x','y'], 'n':[98,2]})
        return pd.DataFrame()


def test_quality_stage():
    ctx = AnalysisContext()
    nn = pd.DataFrame({'column':['a','num1','cat1'], 'non_nulls':[100,100,100], 'pct':[100,100,100]})
    ctx.add_table('profiling.non_null_pct', nn)

    viz = DummyViz()
    QualityStage().run(viz, ctx)

    uniq = ctx.get_table('quality.unique_ratio')
    assert not uniq.empty
    assert 'quasi_constant' in uniq.columns

    catq = ctx.get_table('quality.categorical_quality')
    assert not catq.empty
