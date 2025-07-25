import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pandas as pd
from bq_eda_toolkit.analysis_context import AnalysisContext
from bq_eda_toolkit.stages.validation import ValidationStage

class DummyViz:
    def __init__(self):
        self.schema_df = pd.DataFrame({
            'column_name': ['id', 'name', 'created'],
            'data_type': ['INT64', 'STRING', 'TIMESTAMP'],
            'category': ['numeric', 'string', 'datetime'],
        })
        self.columns = ['id', 'name', 'created']


def test_validation_stage_builds_suite():
    ctx = AnalysisContext()
    viz = DummyViz()
    # schema info
    ctx.add_table('profiling.schema_overview', viz.schema_df)

    # quality tables
    missing = pd.DataFrame({
        'column': ['id', 'name', 'created'],
        'non_nulls': [10, 9, 10],
        'pct': [100.0, 90.0, 100.0],
        'missing_pct': [0.0, 10.0, 0.0],
    })
    ctx.add_table('quality.missing_pct', missing)

    uniq = pd.DataFrame({
        'column': ['id', 'name', 'created'],
        'unique_count': [10, 9, 10],
        'unique_ratio': [100.0, 90.0, 100.0],
        'constant': [False, False, False],
        'quasi_constant': [False, False, False],
    })
    ctx.add_table('quality.unique_ratio', uniq)

    ValidationStage().run(viz, ctx)
    suite_dict = ctx.get_table('validation.expectation_suite')

    assert isinstance(suite_dict, dict)
    expectations = suite_dict.get('expectations', [])
    # ensure each column appears at least once
    cols_in_suite = {e['kwargs']['column'] for e in expectations}
    for col in viz.schema_df['column_name']:
        assert col in cols_in_suite
