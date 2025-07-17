# stages/core_stages.py
from __future__ import annotations
import pandas as pd
import plotly.express as px
from scipy.stats import chi2_contingency
from stages.base import BaseStage
from analysis_context import AnalysisContext
from bigquery_visualizer import BigQueryVisualizer

from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA


# ────────────────────────────────────────────────
# 1. Profiling  (already supplied but kept here for completeness)
# ────────────────────────────────────────────────
class ProfilingStage(BaseStage):
    id = "profiling"

    def run(self, viz: BigQueryVisualizer, ctx: AnalysisContext):
        # schema table already fetched on viz init
        ctx.add_table(self.key("schema_overview"), viz.schema_df.copy())

        # non-null % per column (small query)
        cols = ", ".join(f"COUNTIF({c} IS NOT NULL) AS nn_{c}" for c in viz.columns)
        q = f"SELECT COUNT(*) AS total, {cols} FROM {viz.full_table_path}"
        stats = viz._execute_query(q)
        if stats.empty:
            return
        total = int(stats.at[0, "total"])
        nn_pct = (
            stats.drop(columns="total")
            .melt(var_name="column", value_name="non_nulls")
            .assign(column=lambda d: d["column"].str.replace("nn_", ""),
                    pct=lambda d: d["non_nulls"] / total * 100)
            .sort_values("pct", ascending=False)
        )
        ctx.add_table(self.key("non_null_pct"), nn_pct)

        fig = px.bar(nn_pct, x="column", y="pct",
                     title="Non-null % per column", height=400)
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        ctx.add_figure(self.key("non_null_pct_bar"), fig)


# ────────────────────────────────────────────────
# 2. Data-Quality Stage
# ────────────────────────────────────────────────
class QualityStage(BaseStage):
    id = "quality"

    def run(self, viz: BigQueryVisualizer, ctx: AnalysisContext):
        # Missing-value summary re-uses profiling artefact
        miss_tbl = ctx.get_table("profiling.non_null_pct").copy()
        if miss_tbl is not None:
            miss_tbl["missing_pct"] = 100 - miss_tbl["pct"]
            ctx.add_table(self.key("missing_pct"), miss_tbl)

        # Duplicate row count (all columns)
        dup_q = f"""
          SELECT
            COUNT(*) AS total,
            COUNT(DISTINCT TO_JSON_STRING(t)) AS distinct_rows
          FROM {viz.full_table_path} AS t
        """
        dup = viz._execute_query(dup_q)
        if not dup.empty:
            dup["duplicate_rows"] = dup["total"] - dup["distinct_rows"]
            ctx.add_table(self.key("duplicate_summary"), dup)

        # Outlier rate per numeric column (Tukey rule)
        out_rows = []
        for col in viz.numeric_columns:
            q = f"""
              SELECT
                APPROX_QUANTILES({col}, 4)[OFFSET(1)] AS q1,
                APPROX_QUANTILES({col}, 4)[OFFSET(3)] AS q3,
                COUNT(*) AS n,
                COUNTIF({col} < q1 - 1.5*(q3-q1)
                        OR {col} > q3 + 1.5*(q3-q1)) AS n_out
              FROM {viz.full_table_path}
            """
            row = viz._execute_query(q).iloc[0]
            out_rows.append({"column": col,
                             "outlier_pct": row["n_out"] / row["n"] * 100})
        out_df = pd.DataFrame(out_rows).sort_values("outlier_pct", ascending=False)
        ctx.add_table(self.key("outlier_pct"), out_df)


# ────────────────────────────────────────────────
# 3. Univariate Stage
# ────────────────────────────────────────────────
class UnivariateStage(BaseStage):
    id = "univariate"

    def run(self, viz: BigQueryVisualizer, ctx: AnalysisContext):
        # Numeric descriptive stats (uses existing helper)
        num_df = viz.analyze_all_numeric()
        ctx.add_table(self.key("numeric_stats"), num_df.data if num_df is not None else pd.DataFrame())

        # Categorical descriptive stats (top-5 values)
        cat_df = viz.analyze_all_categorical(top_n_values=5)
        ctx.add_table(self.key("categorical_stats"), cat_df.data if cat_df is not None else pd.DataFrame())

        # Example histogram for first numeric column
        if viz.numeric_columns:
            col = viz.numeric_columns[0]
            _, fig = viz.plot_histogram(numeric_column=col, limit=50000, bins=30)
            ctx.add_figure(self.key(f"{col}.hist"), fig)


# ────────────────────────────────────────────────
# 4. Bivariate Stage
# ────────────────────────────────────────────────
class BivariateStage(BaseStage):
    id = "bivariate"

    def run(self, viz: BigQueryVisualizer, ctx: AnalysisContext):
        # Pearson correlation matrix (sample rows for practicality)
        sample_n = int(ctx.params.get("sample_rows", 200_000))
        num_cols = viz.numeric_columns[:30]                      # cap width
        if num_cols:
            col_list = ", ".join(num_cols)
            q = f"SELECT {col_list} FROM {viz.full_table_path} TABLESAMPLE SYSTEM (1 PERCENT) LIMIT {sample_n}"
            df = viz._execute_query(q)
            corr = df.corr(method="pearson")
            ctx.add_table(self.key("corr_matrix"), corr)

            fig = px.imshow(corr, title="Numeric Correlation Matrix", color_continuous_scale="RdBu_r")
            ctx.add_figure(self.key("corr_heatmap"), fig)

        # χ² for first two categoricals
        if len(viz.categorical_columns) >= 2:
            c1, c2 = viz.categorical_columns[:2]
            q = f"""
                SELECT {c1}, {c2}, COUNT(*) AS n
                FROM {viz.full_table_path}
                WHERE {c1} IS NOT NULL AND {c2} IS NOT NULL
                GROUP BY {c1}, {c2}
            """
            tbl = viz._execute_query(q)
            if not tbl.empty:
                contingency = tbl.pivot_table(index=c1, columns=c2, values="n", fill_value=0)
                chi2, p, *_ = chi2_contingency(contingency)
                ctx.add_table(self.key("chi2_summary"),
                              pd.DataFrame({"chi2": [chi2], "p_value": [p]}))


# ────────────────────────────────────────────────
# 5. Multivariate Stage  (lightweight: VIF & PCA stub)
# ────────────────────────────────────────────────
class MultivariateStage(BaseStage):
    id = "multivar"

    def run(self, viz: BigQueryVisualizer, ctx: AnalysisContext):
        num_cols = viz.numeric_columns[:15]   # keep matrix invertible
        if not num_cols:
            return

        # pull sample
        sample_n = int(ctx.params.get("sample_rows", 100_000))
        q = f"SELECT {', '.join(num_cols)} FROM {viz.full_table_path} TABLESAMPLE SYSTEM (1 PERCENT) LIMIT {sample_n}"
        df = viz._execute_query(q).dropna()
        if df.empty:
            return

        # VIF
        X = df[num_cols].values
        vif = pd.DataFrame({
            "feature": num_cols,
            "VIF": [variance_inflation_factor(X, i) for i in range(X.shape[1])]
        }).sort_values("VIF", ascending=False)
        ctx.add_table(self.key("vif"), vif)

        # Quick PCA (2-D) for visual redundancy check
        pca = PCA(n_components=2).fit_transform(df[num_cols])
        pca_df = pd.DataFrame(pca, columns=["PC1", "PC2"])
        fig = px.scatter(pca_df, x="PC1", y="PC2", opacity=0.4,
                         title="PCA projection (first 2 components)")
        ctx.add_figure(self.key("pca_scatter"), fig)


# ────────────────────────────────────────────────
# 6. Target Diagnostics Stage  (optional)
# ────────────────────────────────────────────────
class TargetStage(BaseStage):
    id = "target"

    def run(self, viz: BigQueryVisualizer, ctx: AnalysisContext):
        target = ctx.params.get("target_column")
        if not target or target not in viz.columns:
            print("TargetStage skipped: no target_column defined.")
            return

        # target distribution
        if target in viz.numeric_columns:
            _, fig = viz.plot_histogram(numeric_column=target, bins=50)
            ctx.add_figure(self.key("target_distribution"), fig)
        else:
            _, fig = viz.plot_pie_chart(dimension=target)
            ctx.add_figure(self.key("target_distribution"), fig)

        # Mutual Information ranking
        sample_n = int(ctx.params.get("sample_rows", 100_000))
        feature_cols = viz.numeric_columns + viz.categorical_columns
        feature_cols = [c for c in feature_cols if c != target][:30]

        if not feature_cols:
            return

        col_sql = ", ".join([target] + feature_cols)
        df = viz._execute_query(f"""
            SELECT {col_sql}
            FROM {viz.full_table_path}
            WHERE {target} IS NOT NULL
            LIMIT {sample_n}
        """).dropna()

        if df.empty:
            return

        X = pd.get_dummies(df[feature_cols], dummy_na=False)
        y = df[target]
        if target in viz.numeric_columns:
            mi = mutual_info_regression(X, y)
        else:
            mi = mutual_info_classif(X, y)
        mi_df = pd.DataFrame({"feature": X.columns, "MI": mi}).sort_values("MI", ascending=False)
        ctx.add_table(self.key("mutual_information"), mi_df)
