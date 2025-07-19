import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


import pandas as pd
from bq_eda_toolkit.analysis_context import AnalysisContext
from bq_eda_toolkit.feature_advice import FeatureAdviceStage

class DummyViz:
    def __init__(self):
        self.full_table_path = 'x'
        self.numeric_columns = ['num1', 'num2']
        self.categorical_columns = ['cat']
        self.datetime_columns = ['date']
        self.columns = self.numeric_columns + self.categorical_columns + self.datetime_columns
        self.queries = []

    def _execute_query(self, q, use_cache=True):
        self.queries.append(q)
        if 'ML.EVALUATE' in q:
            return pd.DataFrame({'metric':['mae'], 'value':[1.0]})
        if 'ML.WEIGHTS' in q:
            return pd.DataFrame({'processed_input':['num1'], 'weight':[0.5]})
        return pd.DataFrame()

def _ctx_with_metrics():
    ctx = AnalysisContext(params={'target_column':'num1'})
    ctx.add_table(
        'quality.missing_pct',
        pd.DataFrame({'column':['num1','num2','cat','date'], 'missing_pct':[10,0,5,60]})
    )
    ctx.add_table(
        'quality.categorical_quality',
        pd.DataFrame({'column':['cat'], 'categories':[3]})
    )
    ctx.add_table(
        'univariate.numeric_stats',
        pd.DataFrame({'Column':['num1','num2'], 'Skewness':[2.0,0.1]})
    )
    corr = pd.DataFrame(
        [[1.0,0.8],[0.8,1.0]],
        index=['num1','num2'],
        columns=['num1','num2']
    )
    ctx.add_table('bivariate.pearson_corr', corr)
    return ctx


def test_feature_advice_outputs():
    ctx = _ctx_with_metrics()
    viz = DummyViz()
    FeatureAdviceStage().run(viz, ctx)

    enc = ctx.get_table('feature_advice.encoding_plan')
    imp = ctx.get_table('feature_advice.imputation_plan')
    scale = ctx.get_table('feature_advice.scaling_plan')
    inter = ctx.get_table('feature_advice.interaction_plan')

    assert {'num1','num2','cat','date'} <= set(enc['column'])
    assert imp.loc[imp['column']=='num1','strategy'].iloc[0] == 'median'
    assert scale.loc[scale['column']=='num2','scaling'].iloc[0] == 'standard'
    assert not inter.empty


def test_feature_advice_bqml_queries_and_storage():
    ctx = _ctx_with_metrics()
    viz = DummyViz()
    FeatureAdviceStage().run(viz, ctx)

    all_q = "\n".join(viz.queries)
    assert 'CREATE OR REPLACE TEMP MODEL' in all_q
    assert 'ML.EVALUATE' in all_q
    assert 'ML.WEIGHTS' in all_q

    assert ctx.get_table('feature_advice.model_metrics') is not None
    assert ctx.get_table('feature_advice.model_weights') is not None
