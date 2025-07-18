from __future__ import annotations
import time
from typing import List, Sequence
from analysis_context import AnalysisContext
from bigquery_visualizer import BigQueryVisualizer

# import all core stages
from stages.core_stages import (
    ProfilingStage,
    QualityStage,
    UnivariateStage,
    BivariateStage,
    MultivariateStage,
    TargetStage,
)
from feature_advice import FeatureAdviceStage

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
    ):
        self.stages = stages or list(DEFAULT_STAGE_CHAIN)
        self.sample_rows = sample_rows
        self.target_column = target_column
        self.verbose = verbose

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
                print(f"\n▶ Stage: {stage.id}")

            try:
                stage.run(viz, ctx)
                elapsed = time.time() - start
                if self.verbose:
                    n_tbl = len([k for k in ctx.tables if k.startswith(stage.id)])
                    n_fig = len([k for k in ctx.figures if k.startswith(stage.id)])
                    print(f"   ✔ finished in {elapsed:0.2f}s"
                          f"  ({n_tbl} tables, {n_fig} figs)")
            except Exception as e:
                print(f"   ⚠️  {stage.id} failed: {e}")

        if self.verbose:
            print("\n✅ Pipeline complete.")
        return ctx