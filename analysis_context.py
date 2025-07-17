# analysis_context.py
from dataclasses import dataclass, field
from typing import Any, Dict
import pandas as pd


@dataclass
class AnalysisContext:
    """
    Shared container passed to every Stage in the pipeline.

    Attributes
    ----------
    tables : dict[str, pd.DataFrame]
        DataFrames produced by each stage (e.g. "profiling.schema_overview").
    figures : dict[str, Any]
        Plotly or Matplotlib objects keyed the same way
        (e.g. "univariate.order_value.hist").
    params : dict[str, Any]
        Run-time parameters you want to keep (sample_rows, target_column, …).
    """
    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    figures: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)

    # convenience helpers
    def add_table(self, key: str, df: pd.DataFrame) -> None:
        self.tables[key] = df

    def add_figure(self, key: str, fig: Any) -> None:
        self.figures[key] = fig

    def get_table(self, key: str) -> pd.DataFrame:
        return self.tables.get(key)

    def get_figure(self, key: str) -> Any:
        return self.figures.get(key)
