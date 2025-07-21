from __future__ import annotations
import pandas as pd
from typing import TYPE_CHECKING
from .base import BaseStage
from ..analysis_context import AnalysisContext

if TYPE_CHECKING:
    from ..bigquery_visualizer import BigQueryVisualizer


class DatasetComparisonStage(BaseStage):
    """Run statistical drift tests between two datasets."""

    id = "comparison"

    def __init__(self, other_viz: 'BigQueryVisualizer', sample_rows: int = 1000, alpha: float = 0.05) -> None:
        self.other_viz = other_viz
        self.sample_rows = sample_rows
        self.alpha = alpha

    def run(self, viz: 'BigQueryVisualizer', ctx: AnalysisContext) -> pd.DataFrame:
        df = viz.compare_to(self.other_viz, sample_rows=self.sample_rows, alpha=self.alpha)
        ctx.add_table(self.key("drift_tests"), df)
        return df
