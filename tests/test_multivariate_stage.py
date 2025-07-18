import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from analysis_context import AnalysisContext
from stages.core_stages import MultivariateStage
from bigquery_visualizer import BigQueryVisualizer

class DummyViz(BigQueryVisualizer):
    def __init__(self):
        self.full_table_path = 'x'
        self.numeric_columns = ['n1', 'n2', 'n3']
        self.categorical_columns = []
        self.columns = self.numeric_columns
    def _execute_query(self, q, use_cache=True):
        return pd.DataFrame({
            'n1':[1,2,3,4,5],
            'n2':[2,4,6,8,10],
            'n3':[5,3,6,2,1]
        })


def test_multivariate_stage_projections():
    ctx = AnalysisContext(params={'sample_rows': 5})
    viz = DummyViz()
    MultivariateStage().run(viz, ctx)

    assert not ctx.get_table('multivar.vif').empty
    assert ctx.get_figure('multivar.pair_plot') is not None
    assert not ctx.get_table('multivar.pca_coords').empty
    assert not ctx.get_table('multivar.tsne_coords').empty
    assert not ctx.get_table('multivar.umap_coords').empty
    metrics = ctx.get_table('multivar.cluster_potential')
    assert not metrics.empty
