from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
from analysis_context import AnalysisContext
from bigquery_visualizer import BigQueryVisualizer   # adjust import path if needed


class BaseStage(ABC):
    """
    Abstract parent for every pipeline stage.

    Sub-classes implement `.run(viz, ctx)` and use `self.key(<suffix>)`
    when storing tables / figures in the AnalysisContext so names remain
    unique and organised (e.g. 'profiling.schema_overview').
    """

    #: short identifier used as prefix in context keys
    id: str = "base"

    def key(self, suffix: str) -> str:
        """Helper to build namespaced keys:  <stage_id>.<suffix>"""
        return f"{self.id}.{suffix}"

    # ↳ concrete stages must implement this
    @abstractmethod
    def run(self, viz: BigQueryVisualizer, ctx: AnalysisContext) -> Any: ...
