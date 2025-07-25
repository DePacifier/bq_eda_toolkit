import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pandas as pd
from bq_eda_toolkit.pipeline import Pipeline
from bq_eda_toolkit.stages.base import BaseStage
from bq_eda_toolkit.analysis_context import AnalysisContext

class DummyViz:
    def refresh_schema(self):
        pass

class DummyStage(BaseStage):
    def __init__(self, stage_id):
        self.id = stage_id
        self.ran = False

    def run(self, viz, ctx):
        self.ran = True
        ctx.add_table(self.key("out"), pd.DataFrame({"x": [self.id]}))


def test_pipeline_runs_all_stages():
    stages = [DummyStage(f"s{i}") for i in range(3)]
    pipe = Pipeline(stages=stages, verbose=False)
    ctx = pipe.run(DummyViz())

    assert isinstance(ctx, AnalysisContext)
    assert all(stage.ran for stage in stages)
    for stage in stages:
        assert ctx.get_table(f"{stage.id}.out") is not None
