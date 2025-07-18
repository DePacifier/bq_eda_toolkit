import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from analysis_context import AnalysisContext
from stages.core_stages import BivariateStage

class DummyViz:
    def __init__(self):
        self.full_table_path = 'x'
        self.numeric_columns = ['n1', 'n2']
        self.categorical_columns = ['cat1', 'cat2']
        self.columns = self.numeric_columns + self.categorical_columns
        self.rep_sample_df = None
    def _execute_query(self, q, use_cache=True):
        if 'CORR(' in q:
            return pd.DataFrame({'c1':['n1'], 'c2':['n2'], 'corr':[0.5]})
        if 'GROUP BY cat1, cat2' in q:
            return pd.DataFrame({'cat1':['A','A','B','B'], 'cat2':['X','Y','X','Y'], 'n':[50,5,5,50]})
        return pd.DataFrame()

    def numeric_correlations(self, columns, method='pearson'):
        return pd.DataFrame([[1.0, 0.5],[0.5,1.0]], index=columns, columns=columns)

    def get_representative_sample(self, columns=None, max_bytes=None, refresh=False):
        if self.rep_sample_df is None or refresh:
            self.rep_sample_df = pd.DataFrame({
                'n1':[1,2,3,4],
                'n2':[10,20,30,40],
                'cat1':['A','B','A','B'],
                'cat2':['X','Y','X','Y']
            })
        df = self.rep_sample_df.copy()
        if columns:
            df = df[columns]
        return df

def test_bivariate_stage_basic():
    ctx = AnalysisContext(params={'lowess_plots': True})
    viz = DummyViz()
    BivariateStage().run(viz, ctx)

    assert not ctx.get_table('bivariate.num_cat_tests').empty
    assert not ctx.get_table('bivariate.cat_pair_tests').empty
    assert not ctx.get_table('bivariate.spearman_corr').empty
    assert any(k.endswith('.lowess') for k in ctx.figures)
    assert ctx.get_figure('bivariate.n1_by_cat1.box') is not None

