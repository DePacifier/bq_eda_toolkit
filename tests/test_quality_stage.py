import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from analysis_context import AnalysisContext
from stages.core_stages import QualityStage

class DummyViz:
    def __init__(self):
        self.full_table_path = 'x'
        self.columns = ['a', 'num1', 'cat1']
        self.numeric_columns = ['num1']
        self.categorical_columns = ['cat1']
        self.rep_sample_df = pd.DataFrame({
            'a':[1,2,3,4],
            'num1':[1,2,3,100],
            'cat1':['x','y','x','z']
        })
    def _execute_query(self, q, use_cache=True):
        if 'TO_JSON_STRING' in q:
            return pd.DataFrame({'total':[100], 'distinct_rows':[95]})
        if 'COUNT(DISTINCT' in q and 'uniq_' in q:
            return pd.DataFrame({'total':[100], 'uniq_a':[3], 'uniq_num1':[90], 'uniq_cat1':[2]})
        if 'n_out_z' in q or 'n_out_iqr' in q:
            return pd.DataFrame({'q1':[0],'q3':[1],'mean':[0.5],'sd':[1],'n':[100],'n_out_iqr':[5],'n_out_z':[4]})
        if 'APPROX_QUANTILES' in q:
            return pd.DataFrame({'q1':[0],'q3':[1],'n':[100],'n_out':[5]})
        if 'GROUP BY cat1' in q:
            return pd.DataFrame({'cat1':['x','y','X'], 'n':[97,2,1]})
        if 'TABLESAMPLE' in q:
            return pd.DataFrame({'num1':[1,2,3,100]})
        return pd.DataFrame()

    def get_representative_sample(self, columns=None, max_bytes=None, refresh=False):
        df = self.rep_sample_df.copy()
        if columns:
            df = df[columns]
        return df

    def missingness_map(self, columns=None, sample_rows=100000):
        df = pd.DataFrame({'a':[1,2], 'num1':[1,2], 'cat1':['x','y']})
        mask = df.isna()
        fig, ax = plt.subplots()
        sns.heatmap(mask.T, cbar=False, ax=ax)
        mcar = pd.DataFrame({'column': self.columns, 'MCAR': [True]*len(self.columns)})
        return mask, ax, mcar


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

    out = ctx.get_table('quality.outlier_pct')
    assert 'zscore_outlier_pct' in out.columns

    iso = ctx.get_table('quality.outlier_flags')
    assert iso is not None

    inc = ctx.get_table('quality.cat1.inconsistent_groups')
    assert not inc.empty

    assert 'quality.missing_map' in ctx.figures
    mcar = ctx.get_table('quality.mcar_results')
    assert not mcar.empty
