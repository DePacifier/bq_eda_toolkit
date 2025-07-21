from __future__ import annotations

from typing import TYPE_CHECKING

from .base import BaseStage
from ..analysis_context import AnalysisContext
from ..validation import build_expectation_suite

if TYPE_CHECKING:  # pragma: no cover
    from ..bigquery_visualizer import BigQueryVisualizer


class ValidationStage(BaseStage):
    """Generate a Great Expectations suite from prior stage outputs."""

    id = "validation"

    def run(self, viz: 'BigQueryVisualizer', ctx: AnalysisContext):
        suite = build_expectation_suite(viz, ctx)
        ctx.tables[self.key("expectation_suite")] = suite.to_json_dict()
        return suite
