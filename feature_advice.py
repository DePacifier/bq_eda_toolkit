from __future__ import annotations
import pandas as pd
from .stages.base import BaseStage
from .analysis_context import AnalysisContext
from .bigquery_visualizer import BigQueryVisualizer


class FeatureAdviceStage(BaseStage):
    """Suggest encoding, scaling and imputation strategies."""

    id = "feature_advice"

    def run(self, viz: BigQueryVisualizer, ctx: AnalysisContext):
        missing = ctx.get_table("quality.missing_pct")
        cat_quality = ctx.get_table("quality.categorical_quality")
        num_stats = ctx.get_table("univariate.numeric_stats")
        corr = ctx.get_table("bivariate.pearson_corr")

        enc_rows, imp_rows, scale_rows, inter_rows = [], [], [], []

        def missing_pct(col: str) -> float:
            if missing is None or missing.empty:
                return 0.0
            row = missing[missing["column"] == col]
            return float(row["missing_pct"].iloc[0]) if not row.empty else 0.0

        def skew(col: str) -> float:
            if num_stats is None or num_stats.empty:
                return 0.0
            row = num_stats[num_stats["Column"] == col]
            if row.empty:
                row = num_stats[num_stats["column"] == col]
            return float(row["Skewness"].iloc[0]) if not row.empty else 0.0

        def n_categories(col: str) -> int:
            if cat_quality is None or cat_quality.empty:
                return 0
            row = cat_quality[cat_quality["column"] == col]
            return int(row["categories"].iloc[0]) if not row.empty else 0

        # --- encoding + imputation + scaling suggestions ---
        for col in viz.numeric_columns:
            mp = missing_pct(col)
            sk = skew(col)
            enc_rows.append({"column": col, "encoding": "numeric"})
            if mp > 0:
                imp_rows.append({"column": col, "strategy": "median" if abs(sk) > 1 else "mean"})
            else:
                imp_rows.append({"column": col, "strategy": "none"})
            scale_rows.append({"column": col, "scaling": "log" if abs(sk) > 1 else "standard"})

        for col in viz.categorical_columns:
            cats = n_categories(col)
            mp = missing_pct(col)
            encoding = "one-hot" if cats and cats <= 20 else "target"
            enc_rows.append({"column": col, "encoding": encoding})
            strategy = "mode" if mp < 30 else "constant('missing')"
            imp_rows.append({"column": col, "strategy": strategy})

        for col in viz.datetime_columns:
            enc_rows.append({"column": col, "encoding": "datetime"})
            mp = missing_pct(col)
            imp_rows.append({"column": col, "strategy": "drop" if mp > 50 else "none"})
            inter_rows.append({"feature_1": col, "feature_2": "", "suggestion": "lag_features"})

        # --- interaction ideas based on correlation ---
        if corr is not None and not corr.empty:
            for i, col1 in enumerate(viz.numeric_columns):
                for col2 in viz.numeric_columns[i+1:]:
                    if col1 in corr.index and col2 in corr.columns:
                        val = corr.loc[col1, col2]
                    elif col2 in corr.index and col1 in corr.columns:
                        val = corr.loc[col2, col1]
                    else:
                        continue
                    if pd.notna(val) and abs(val) > 0.7:
                        inter_rows.append({"feature_1": col1, "feature_2": col2, "suggestion": "polynomial"})

        ctx.add_table(self.key("encoding_plan"), pd.DataFrame(enc_rows))
        ctx.add_table(self.key("imputation_plan"), pd.DataFrame(imp_rows))
        ctx.add_table(self.key("scaling_plan"), pd.DataFrame(scale_rows))
        ctx.add_table(self.key("interaction_plan"), pd.DataFrame(inter_rows))
