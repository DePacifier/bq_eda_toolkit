from __future__ import annotations
import time
from typing import List, Sequence
import logging
from .analysis_context import AnalysisContext
from .bigquery_visualizer import BigQueryVisualizer

# import all core stages
from .stages.core_stages import (
    ProfilingStage,
    QualityStage,
    UnivariateStage,
    BivariateStage,
    MultivariateStage,
    TargetStage,
)
from .feature_advice import FeatureAdviceStage

logger = logging.getLogger(__name__)

# default chain (order matters)
DEFAULT_STAGE_CHAIN: Sequence = [
    ProfilingStage(),
    QualityStage(),
    UnivariateStage(),
    BivariateStage(),
    MultivariateStage(),
    TargetStage(),
    FeatureAdviceStage(),
]


class Pipeline:
    """
    Lightweight orchestrator that executes Stage objects in sequence.

    Parameters
    ----------
    stages : list[BaseStage] | None
        Custom stage order; None uses DEFAULT_STAGE_CHAIN.
    sample_rows : int
        Row cap for sample-based stages (corr, VIF, MI, …).
    target_column : str | None
        Provide if you want the TargetDiagnostics stage to run.
    verbose : bool
        Print stage timing and summary keys.
    """

    def __init__(
        self,
        stages: List = None,
        *,
        sample_rows: int = 200_000,
        target_column: str | None = None,
        verbose: bool = True,
        fail_fast: bool = False,
    ):
        self.stages = stages or list(DEFAULT_STAGE_CHAIN)
        self.sample_rows = sample_rows
        self.target_column = target_column
        self.verbose = verbose
        self.fail_fast = fail_fast

    # ──────────────────────────────────────────────────────────
    # public API
    # ──────────────────────────────────────────────────────────
    def run(self, viz: BigQueryVisualizer) -> AnalysisContext:
        """
        Execute all stages and return an AnalysisContext holding
        every DataFrame / Figure produced.
        """
        # Refresh schema in case table changed mid-notebook
        viz.refresh_schema()

        ctx = AnalysisContext(
            params={
                "sample_rows": self.sample_rows,
                "target_column": self.target_column,
            }
        )

        for stage in self.stages:
            start = time.time()
            if self.verbose:
                logger.info("Stage: %s", stage.id)

            try:
                stage.run(viz, ctx)
                elapsed = time.time() - start
                if self.verbose:
                    n_tbl = len([k for k in ctx.tables if k.startswith(stage.id)])
                    n_fig = len([k for k in ctx.figures if k.startswith(stage.id)])
                    logger.info(
                        "finished in %0.2fs  (%d tables, %d figs)",
                        elapsed,
                        n_tbl,
                        n_fig,
                    )
            except Exception as e:
                if self.fail_fast:
                    raise
                logger.warning("%s failed: %s", stage.id, e)

        if self.verbose:
            logger.info("Pipeline complete.")
        return ctx
