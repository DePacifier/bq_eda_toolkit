import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


import pandas as pd
from bq_eda_toolkit.analysis_context import AnalysisContext
from bq_eda_toolkit.stages.core_stages import UnivariateStage
from bq_eda_toolkit.bigquery_visualizer import BigQueryVisualizer

class DummyViz(BigQueryVisualizer):
    def __init__(self):
        self.full_table_path = 'x'
        self.numeric_columns = ['num1', 'num2']
        self.columns = self.numeric_columns
        self.categorical_columns = []
        self.auto_show = False
    def _execute_query(self, q, use_cache=True):
        if 'VAR_SAMP(num1)' in q:
            return pd.DataFrame({
                'total_rows':[4],'null_count':[0],'non_null_count':[4],
                'mean':[2.5],'std_dev':[1.12],'variance':[1.25],'min':[1],'max':[4],
                'skewness':[0.0],'kurtosis':[1.5],'quartiles':[[1,1.75,2.5,3.25,4]]})
        if 'VAR_SAMP(num2)' in q:
            return pd.DataFrame({
                'total_rows':[4],'null_count':[0],'non_null_count':[4],
                'mean':[25],'std_dev':[11.2],'variance':[125.0],'min':[10],'max':[40],
                'skewness':[0.0],'kurtosis':[1.5],'quartiles':[[10,17.5,25,32.5,40]]})
        if 'APPROX_QUANTILES' in q:
            return pd.DataFrame({
                'bucket':[1,2],
                'bin_start':[0,5],
                'bin_end':[5,10],
                'n':[2,2]
            })
        return pd.DataFrame()

    def get_representative_sample(self, columns=None, max_bytes=None, refresh=False):
        df = pd.DataFrame({'num1':[1,2,3,4], 'num2':[10,20,30,40]})
        if columns:
            df = df[columns]
        return df


def test_univariate_stage_kde_histograms():
    ctx = AnalysisContext()
    viz = DummyViz()
    UnivariateStage().run(viz, ctx)

    stats = ctx.get_table('univariate.numeric_stats')
    assert 'Skewness' in stats.columns
    assert len([k for k in ctx.figures if k.endswith('.hist')]) == len(viz.numeric_columns)
    for col in viz.numeric_columns:
        fig = ctx.get_figure(f'univariate.{col}.hist')
        assert fig is not None
        assert len(fig.data) > 1
