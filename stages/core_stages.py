# stages/core_stages.py
from __future__ import annotations
import pandas as pd
import plotly.express as px
from scipy.stats import chi2_contingency, ttest_ind, f_oneway
from scipy.stats import spearmanr
import numpy as np
from typing import TYPE_CHECKING
from .base import BaseStage
from ..analysis_context import AnalysisContext
if TYPE_CHECKING:
    from ..bigquery_visualizer import BigQueryVisualizer

from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest


# ────────────────────────────────────────────────
# Representative Sampling Stage
# ────────────────────────────────────────────────
class RepSampleStage(BaseStage):
    """Return a small, representative subset of the table."""

    id = "rep_sample"

    def __init__(
        self,
        *,
        n: int | None = None,
        sample_percent: float | None = None,
        method: str = "TABLESAMPLE",
        stratify_by: str | None = None,
        columns: list[str] | None = None,
    ) -> None:
        self.n = n
        self.sample_percent = sample_percent
        self.method = method.upper()
        self.stratify_by = stratify_by
        self.columns = columns

    # rough byte sizes per column type
    _SIZE_MAP = {
        "numeric": 8,
        "string": 20,
        "boolean": 1,
        "datetime": 8,
        "complex": 50,
        "geographic": 16,
        "other": 8,
    }

    def _row_bytes(self, viz: BigQueryVisualizer) -> int:
        cols = self.columns or viz.columns
        cat_lookup = dict(zip(viz.schema_df["column_name"], viz.schema_df["category"]))
        row_bytes = sum(self._SIZE_MAP.get(cat_lookup.get(c, "other"), 8) for c in cols)
        return max(1, row_bytes)

    def _estimate_n(self, viz: BigQueryVisualizer) -> int:
        if self.n is not None:
            return int(self.n)
        limit_bytes = getattr(viz, "max_result_bytes", 0) or 0
        n = limit_bytes // self._row_bytes(viz)
        return max(1, int(n))

    def _derive_percent(self, viz: BigQueryVisualizer, n: int) -> float:
        if self.sample_percent is not None:
            return float(self.sample_percent)
        if getattr(viz, "table_rows", None):
            pct = n / viz.table_rows * 100
            return pct
        return 1.0

    def build_query(self, viz: BigQueryVisualizer) -> str:
        cols = self.columns or viz.columns
        col_sql = ", ".join(cols)
        n = self._estimate_n(viz)
        pct = self._derive_percent(viz, n)
        pct_str = f"{pct:g}"
        if self.method == "STRATIFIED":
            if not self.stratify_by:
                raise ValueError("stratify_by required for STRATIFIED sampling")
            return (
                f"WITH base AS (\n"
                f"  SELECT {col_sql},\n"
                f"         ROW_NUMBER() OVER(PARTITION BY {self.stratify_by} ORDER BY RAND()) AS rn,\n"
                f"         COUNT(*) OVER(PARTITION BY {self.stratify_by}) AS cnt\n"
                f"  FROM {viz.full_table_path}\n"
                f")\n"
                f"SELECT {col_sql}\n"
                f"FROM base\n"
                f"WHERE rn <= CEIL(cnt * {pct_str} / 100)\n"
                f"LIMIT {n}"
            )
        # default TABLESAMPLE
        return (
            f"SELECT {col_sql} FROM {viz.full_table_path} "
            f"TABLESAMPLE SYSTEM ({pct_str} PERCENT) "
            f"LIMIT {n}"
        )

    def run(self, viz: BigQueryVisualizer, ctx: AnalysisContext) -> pd.DataFrame:
        q = self.build_query(viz)
        df = viz._execute_query(q)
        ctx.add_table(self.key("sample"), df)
        return df


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

        # Sample bias check
        sample_n = int(ctx.params.get("sample_rows", 1000))
        bias_df = viz.evaluate_sample_bias(sample_rows=sample_n)
        ctx.add_table(self.key("sample_bias"), bias_df)


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

        # Missingness map & MCAR/MAR tests
        mask_df, ax, mcar_df = viz.missingness_map()
        if not mask_df.empty:
            if ax is not None:
                ctx.add_figure(self.key("missing_map"), ax.get_figure())
            ctx.add_table(self.key("mcar_results"), mcar_df)

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

        # Unique value ratios
        uniq_cols = ", ".join(
            f"COUNT(DISTINCT {c}) AS uniq_{c}" for c in viz.columns
        )
        uniq_q = f"SELECT COUNT(*) AS total, {uniq_cols} FROM {viz.full_table_path}"
        uniq = viz._execute_query(uniq_q)
        if not uniq.empty:
            total = int(uniq.at[0, 'total'])
            rows = []
            for c in viz.columns:
                u = uniq.at[0, f'uniq_{c}']
                ratio = u / total if total else 0
                rows.append({
                    'column': c,
                    'unique_count': u,
                    'unique_ratio': ratio * 100,
                    'constant': u <= 1,
                    'quasi_constant': ratio < 0.05 and u > 1,
                })
            uniq_df = pd.DataFrame(rows)
            ctx.add_table(self.key("unique_ratio"), uniq_df)

        # Categorical value counts
        cat_summary = []
        for col in viz.categorical_columns:
            counts = viz._execute_query(
                f"SELECT {col}, COUNT(*) as n FROM {viz.full_table_path} GROUP BY {col}"
            )
            if counts.empty:
                continue
            ctx.add_table(self.key(f"{col}.category_counts"), counts)
            n_cats = len(counts)
            n_single = int((counts['n'] == 1).sum())
            ratio_single = n_single / n_cats if n_cats else 0
            long_tail = n_cats > 50 or ratio_single > 0.5
            cat_summary.append({
                'column': col,
                'categories': n_cats,
                'singleton_pct': ratio_single * 100,
                'long_tail': long_tail,
            })
        if cat_summary:
            ctx.add_table(self.key("categorical_quality"), pd.DataFrame(cat_summary))

        # Detect inconsistent categorical values (case/whitespace)
        for col in viz.categorical_columns:
            counts = ctx.get_table(self.key(f"{col}.category_counts"))
            if counts is None or counts.empty:
                continue
            norm = counts[col].astype(str).str.strip().str.lower()
            groups: dict[str, set] = {}
            for orig, nval in zip(counts[col].astype(str), norm):
                groups.setdefault(nval, set()).add(orig)
            inconsist = {k: list(v) for k, v in groups.items() if len(v) > 1}
            if inconsist:
                df_inc = pd.DataFrame({
                    "normalised": list(inconsist.keys()),
                    "original_values": list(inconsist.values()),
                })
                ctx.add_table(self.key(f"{col}.inconsistent_groups"), df_inc)

        # Outlier rate per numeric column (Tukey, Z-score) and IsolationForest
        out_rows = []
        for col in viz.numeric_columns:
            q = f"""
              WITH stats AS (
                SELECT
                  APPROX_QUANTILES({col}, 4)[OFFSET(1)] AS q1,
                  APPROX_QUANTILES({col}, 4)[OFFSET(3)] AS q3,
                  AVG({col}) AS mean,
                  STDDEV_POP({col}) AS sd
                FROM {viz.full_table_path}
              )
              SELECT
                q1, q3, mean, sd,
                COUNT(*) AS n,
                COUNTIF({col} < q1 - 1.5*(q3 - q1)
                        OR {col} > q3 + 1.5*(q3 - q1)) AS n_out_iqr,
                COUNTIF(ABS(({col} - mean)/NULLIF(sd,0)) > 3) AS n_out_z
              FROM {viz.full_table_path}, stats
            """
            row = viz._execute_query(q).iloc[0]
            out_rows.append({
                "column": col,
                "iqr_outlier_pct": row["n_out_iqr"] / row["n"] * 100,
                "zscore_outlier_pct": row["n_out_z"] / row["n"] * 100,
            })
        out_df = pd.DataFrame(out_rows).sort_values("iqr_outlier_pct", ascending=False)
        ctx.add_table(self.key("outlier_pct"), out_df)

        # IsolationForest on numeric sample
        sample_n = int(ctx.params.get("sample_rows", 10000))
        if viz.numeric_columns:
            sample_df = (
                viz.get_representative_sample(columns=viz.numeric_columns)
                .collect()
                .to_pandas()
            )
            if len(sample_df) > sample_n:
                sample_df = sample_df.sample(sample_n, random_state=42)
            if not sample_df.empty:
                iso = IsolationForest(contamination=0.01, random_state=42)
                X = sample_df[viz.numeric_columns].dropna()
                if not X.empty:
                    flags = iso.fit_predict(X)
                    pct = (flags == -1).mean() * 100
                    ctx.add_table(
                        self.key("outlier_flags"),
                        pd.DataFrame({"method": ["IsolationForest"], "flagged_pct": [pct]}),
                    )


# ────────────────────────────────────────────────
# 3. Univariate Stage
# ────────────────────────────────────────────────
class UnivariateStage(BaseStage):
    id = "univariate"

    def run(self, viz: BigQueryVisualizer, ctx: AnalysisContext):
        # Numeric descriptive stats (uses existing helper)
        num_df = viz.analyze_all_numeric()
        if hasattr(num_df, 'data'):
            ctx.add_table(self.key("numeric_stats"), num_df.data)
        else:
            ctx.add_table(self.key("numeric_stats"), num_df if num_df is not None else pd.DataFrame())

        # Categorical descriptive stats (top-5 values)
        cat_df = viz.analyze_all_categorical(top_n_values=5)
        if hasattr(cat_df, 'data'):
            ctx.add_table(self.key("categorical_stats"), cat_df.data)
        else:
            ctx.add_table(self.key("categorical_stats"), cat_df if cat_df is not None else pd.DataFrame())

        # Histograms with KDE overlay for each numeric column
        for col in viz.numeric_columns:
            _, fig = viz.plot_histogram(numeric_column=col, limit=50000,
                                       bins=30, kde=True)
            ctx.add_figure(self.key(f"{col}.hist"), fig)


# ────────────────────────────────────────────────
# 4. Bivariate Stage
# ────────────────────────────────────────────────
class BivariateStage(BaseStage):
    id = "bivariate"

    def run(self, viz: BigQueryVisualizer, ctx: AnalysisContext):
        sample_n = int(ctx.params.get("sample_rows", 200_000))
        num_cols = viz.numeric_columns[:30]

        # ─── Numeric correlations ──────────────────────────────────────
        if num_cols:
            pear = viz.numeric_correlations(num_cols, method="pearson")
            if not pear.empty:
                ctx.add_table(self.key("pearson_corr"), pear)
                fig = px.imshow(pear, title="Numeric Correlation Matrix", color_continuous_scale="RdBu_r")
                ctx.add_figure(self.key("corr_heatmap"), fig)

            spear = viz.numeric_correlations(num_cols, method="spearman")
            if not spear.empty:
                ctx.add_table(self.key("spearman_corr"), spear)

            if ctx.params.get("lowess_plots"):
                from itertools import combinations
                df = (
                    viz.get_representative_sample(columns=num_cols[:5])
                    .collect()
                    .to_pandas()
                )
                for x, y in combinations(num_cols[:5], 2):
                    if df.empty:
                        break
                    fig = px.scatter(df, x=x, y=y, trendline="lowess",
                                     title=f"{y} vs {x} (LOWESS)")
                    ctx.add_figure(self.key(f"{y}_vs_{x}.lowess"), fig)

        # ─── Numeric ~ Categorical tests ───────────────────────────────
        numcat_results = []
        for num in viz.numeric_columns:
            for cat in viz.categorical_columns:
                df_pair = (
                    viz.get_representative_sample(columns=[cat, num])
                    .collect()
                    .to_pandas()
                ).dropna()
                if len(df_pair) > sample_n:
                    df_pair = df_pair.sample(sample_n, random_state=42)
                if df_pair.empty:
                    continue

                fig = px.box(df_pair, x=cat, y=num, points="outliers",
                             title=f"{num} by {cat}")
                ctx.add_figure(self.key(f"{num}_by_{cat}.box"), fig)

                groups = [g[num].tolist() for _, g in df_pair.groupby(cat)]
                if len(groups) < 2:
                    continue
                if len(groups) == 2:
                    _, p = ttest_ind(groups[0], groups[1], equal_var=False)
                    test = "t-test"
                else:
                    _, p = f_oneway(*groups)
                    test = "anova"
                numcat_results.append({
                    "numeric_column": num,
                    "categorical_column": cat,
                    "test": test,
                    "p_value": p,
                })

        if numcat_results:
            ctx.add_table(self.key("num_cat_tests"), pd.DataFrame(numcat_results))

        # ─── Categorical pair tests ────────────────────────────────────
        cat_results = []
        from itertools import combinations
        for c1, c2 in combinations(viz.categorical_columns, 2):
            q = f"""
                SELECT {c1}, {c2}, COUNT(*) AS n
                FROM {viz.full_table_path}
                WHERE {c1} IS NOT NULL AND {c2} IS NOT NULL
                GROUP BY {c1}, {c2}
            """
            tbl = viz._execute_query(q)
            if tbl.empty:
                continue
            contingency = tbl.pivot_table(index=c1, columns=c2, values="n", fill_value=0)
            chi2, p, _, _ = chi2_contingency(contingency)
            if p >= 0.05:
                continue
            n = contingency.values.sum()
            r, k = contingency.shape
            denom = n * (min(r - 1, k - 1))
            cramers_v = np.sqrt(chi2 / denom) if denom else np.nan
            cat_results.append({
                "column_1": c1,
                "column_2": c2,
                "chi2": chi2,
                "p_value": p,
                "cramers_v": cramers_v,
            })

        if cat_results:
            ctx.add_table(self.key("cat_pair_tests"), pd.DataFrame(cat_results))


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
        df = (
            viz.get_representative_sample(columns=num_cols)
            .collect()
            .to_pandas()
        ).dropna()
        if len(df) > sample_n:
            df = df.sample(sample_n, random_state=42)
        if df.empty:
            return

        # VIF
        X = df[num_cols].values
        vif = pd.DataFrame({
            "feature": num_cols,
            "VIF": [variance_inflation_factor(X, i) for i in range(X.shape[1])]
        }).sort_values("VIF", ascending=False)
        ctx.add_table(self.key("vif"), vif)

        # Scatter matrix on a lightweight sample
        _, pair_fig = viz.pair_plot(num_cols[:5], sample_rows=min(2000, sample_n))
        if pair_fig is not None:
            ctx.add_figure(self.key("pair_plot"), pair_fig)

        # ─── 2-D projections ───────────────────────────────────────
        pca_df, pca_fig = viz.project_2d(method="pca", columns=num_cols, sample_rows=sample_n)
        if not pca_df.empty:
            ctx.add_table(self.key("pca_coords"), pca_df)
            ctx.add_figure(self.key("pca_scatter"), pca_fig)

        tsne_df, tsne_fig = viz.project_2d(method="tsne", columns=num_cols, sample_rows=sample_n)
        if not tsne_df.empty:
            ctx.add_table(self.key("tsne_coords"), tsne_df)
            ctx.add_figure(self.key("tsne_scatter"), tsne_fig)

        umap_df, umap_fig = viz.project_2d(method="umap", columns=num_cols, sample_rows=sample_n)
        if not umap_df.empty:
            ctx.add_table(self.key("umap_coords"), umap_df)
            ctx.add_figure(self.key("umap_scatter"), umap_fig)

        # ─── Clustering potential metrics ───────────────────────────
        hopkins = viz.hopkins_statistic(X)
        sil = viz.silhouette_score_estimate(X)
        ctx.add_table(self.key("cluster_potential"),
                      pd.DataFrame({"hopkins_stat": [hopkins], "silhouette_score": [sil]}))


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

        # class distribution
        cls_df = viz._execute_query(
            f"SELECT {target} AS class, COUNT(*) as n FROM {viz.full_table_path} GROUP BY {target}"
        )
        if not cls_df.empty:
            total = cls_df["n"].sum()
            cls_df["pct"] = cls_df["n"] / total * 100
            ctx.add_table(self.key("class_balance"), cls_df.sort_values("pct", ascending=False))
