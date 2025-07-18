import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from google.cloud import bigquery
from google.oauth2 import service_account
from pathlib import Path
import hashlib
import numpy as np
from scipy.stats import ks_2samp, chi2_contingency, gaussian_kde
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BigQueryVisualizer:
    """
    An enhanced class to connect to a BigQuery table and generate visualizations
    and descriptive analyses directly in a Python notebook.
    """
    def __init__(
        self,
        project_id: str,
        table_id: str,
        credentials_path: str = None,
        max_bytes_scanned: int = 10_000_000_000,
        max_result_bytes: int = 2_000_000_000,
        cache_threshold_bytes: int = 100_000_000,
        sample_cache_dir: str = ".rep_samples",
    ):
        """
        Initializes the visualizer with BigQuery credentials and table info.

        Args:
            project_id (str): Your Google Cloud project ID.
            table_id (str): The full ID of the BigQuery table (e.g., 'your_dataset.your_table_name').
            credentials_path (str, optional): Path to your GCP service account JSON file.
                                              If None, relies on default authentication.
            max_bytes_scanned (int): Abort if a dry run estimates scanning more than this many bytes.
            max_result_bytes (int): Abort if ``EXPLAIN`` estimates results larger than this many bytes.
            cache_threshold_bytes (int): Only cache DataFrames smaller than this size.
            sample_cache_dir (str): Directory to persist representative samples.
        """
        self.project_id = project_id
        self.table_id = table_id
        self.full_table_path = f"`{self.project_id}.{self.table_id}`"
        self._query_cache: dict[str, pd.DataFrame] = {}
        self.max_bytes_scanned = max_bytes_scanned
        self.max_result_bytes = max_result_bytes
        self.cache_threshold_bytes = cache_threshold_bytes
        self.rep_sample_df: pd.DataFrame | None = None
        self.rep_sample_columns_key: str | None = None
        self.sample_cache_dir = Path(sample_cache_dir)
        self.sample_cache_dir.mkdir(parents=True, exist_ok=True)
        
        if credentials_path:
            self.credentials = service_account.Credentials.from_service_account_file(credentials_path)
            self.client = bigquery.Client(credentials=self.credentials, project=self.project_id)
        else:
            self.client = bigquery.Client(project=project_id)
            
        logger.info("Fetching detailed table schema...")
        # Initialize properties for each data type category
        self.refresh_schema()

        # ──────────────── fetch basic table stats ────────────────
        dataset_id, table_name = self.table_id.split('.')
        meta = self._execute_query(
            f"""
            SELECT row_count, size_bytes
            FROM `{dataset_id}.INFORMATION_SCHEMA.TABLES`
            WHERE table_name = '{table_name}'
            """
        )
        if not meta.empty:
            self.table_rows = int(meta['row_count'].iloc[0])
            self.table_size_gb = float(meta['size_bytes'].iloc[0]) / 1e9
        else:
            self.table_rows = None
            self.table_size_gb = None

        logger.info(f"✅ BigQueryVisualizer initialized for table: {self.table_id}")
        logger.info(f"    ℹ️ Found {len(self.columns)} total columns:")
        if self.numeric_columns:
            logger.info(f"     - {len(self.numeric_columns)} numeric")
        if self.string_columns:
            logger.info(f"     - {len(self.string_columns)} string")
        if self.boolean_columns:
            logger.info(f"     - {len(self.boolean_columns)} boolean")
        if self.datetime_columns:
            logger.info(f"     - {len(self.datetime_columns)} datetime")
        if self.complex_columns:
            logger.info(f"     - {len(self.complex_columns)} complex")
        if self.geographic_columns:
            logger.info(f"     - {len(self.geographic_columns)} geographic")
        if self.table_rows is not None and self.table_size_gb is not None:
            logger.info(
                f"    ℹ️ {self.table_rows:,} rows · {self.table_size_gb:.2f} GB"
            )

    def refresh_schema(self):
        """Re-fetch INFORMATION_SCHEMA and update column lists."""
        self.schema_df = self._fetch_and_categorize_schema()
        self.columns          = self.schema_df['column_name'].tolist()
        self.numeric_columns  = self.schema_df[self.schema_df['category'] == 'numeric']['column_name'].tolist()
        self.string_columns   = self.schema_df[self.schema_df['category'] == 'string']['column_name'].tolist()
        self.boolean_columns  = self.schema_df[self.schema_df['category'] == 'boolean']['column_name'].tolist()
        self.datetime_columns = self.schema_df[self.schema_df['category'] == 'datetime']['column_name'].tolist()
        self.complex_columns  = self.schema_df[self.schema_df['category'] == 'complex']['column_name'].tolist()
        self.geographic_columns = self.schema_df[self.schema_df['category'] == 'geographic']['column_name'].tolist()
        self.categorical_columns = self.string_columns + self.boolean_columns

    def _fetch_and_categorize_schema(self) -> pd.DataFrame:
        """Helper to retrieve a detailed schema using INFORMATION_SCHEMA."""
        dataset_id, table_name = self.table_id.split('.')
        
        query = f"""
            SELECT
              column_name,
              data_type,
              CASE
                WHEN data_type IN ('INT64', 'FLOAT64', 'NUMERIC', 'BIGNUMERIC') THEN 'numeric'
                WHEN data_type IN ('STRING', 'BYTES') THEN 'string'
                WHEN data_type IN ('BOOL') THEN 'boolean'
                WHEN data_type IN ('DATE', 'DATETIME', 'TIMESTAMP', 'TIME') THEN 'datetime'
                WHEN data_type IN ('ARRAY', 'STRUCT') THEN 'complex'
                WHEN data_type IN ('GEOGRAPHY') THEN 'geographic'
                ELSE 'other'
              END as category
            FROM `{dataset_id}.INFORMATION_SCHEMA.COLUMNS`
            WHERE table_name = '{table_name}'
            ORDER BY ordinal_position;
        """
        return self._execute_query(query)

    def _execute_query(self, query: str, use_cache: bool = True) -> pd.DataFrame:
        """Helper function to execute a query and return a DataFrame."""
        if use_cache and query in self._query_cache:
            return self._query_cache[query].copy()

        cfg = bigquery.QueryJobConfig(dry_run=True, use_query_cache=True)
        dry_job = self.client.query(query, job_config=cfg)
        if dry_job.total_bytes_processed > self.max_bytes_scanned:
            raise RuntimeError(
                f"Query would process {dry_job.total_bytes_processed/1e9:.2f} GB "
                f"(limit {self.max_bytes_scanned/1e9:.2f} GB). Aborting."
            )

        # Estimate result size before executing
        est_bytes = 0
        try:
            plan_job = self.client.query(f"EXPLAIN {query}")
            plan_job.result()
            if plan_job.query_plan:
                final = plan_job.query_plan[-1]
                est_bytes = int(
                    final._properties.get("statistics", {}).get("estimatedBytes", 0)
                )
        except Exception:
            est_bytes = 0

        if est_bytes and est_bytes > self.max_result_bytes:
            raise RuntimeError(
                f"Query would return {est_bytes/1e9:.2f} GB "
                f"(limit {self.max_result_bytes/1e9:.2f} GB). Aborting."
            )

        logger.info(
            "ℹ️ Query will process %0.2f GB",
            dry_job.total_bytes_processed / 1e9,
        )

        try:
            query_job = self.client.query(query)
            df = query_job.to_dataframe()
            if use_cache and df.memory_usage(deep=True).sum() < self.cache_threshold_bytes:
                self._query_cache[query] = df.copy()
            return df
        except Exception as e:
            logger.warning("An error occurred: %s", e)
            return pd.DataFrame()
        
    def clear_cache(self):
        """Empty the in‑memory query cache."""
        self._query_cache.clear()

    def _build_where_clause(self, filter_str: str) -> str:
        """Helper to build a SQL WHERE clause from a filter string."""
        return f"WHERE {filter_str}" if filter_str else ""

    def _null_filter_for_dims(self, dims: list[str]) -> str:
        """Return 'col1 IS NOT NULL AND col2 IS NOT NULL …' or ''."""
        return " AND ".join(f"{d} IS NOT NULL" for d in dims)

    def _merge_where(self, base_where: str, extra: str) -> str:
        """
        Combine two WHERE fragments safely.
        `base_where` is "" or starts with "WHERE".
        `extra` is a raw condition without leading WHERE / AND.
        """
        if not extra:
            return base_where
        if not base_where.strip():
            return f"WHERE {extra}"
        return f"{base_where} AND {extra}"
        
    def display_table(self, columns: list = None, order_by: str = None, limit: int = 25):
        """
        Displays data from the table in a formatted way.

        Args:
            columns (list, optional): A list of column names to display. Defaults to all ('*').
            order_by (str, optional): Column to sort the results by (e.g., 'column_name DESC').
            limit (int): The number of rows to display.

        Returns:
            pandas.io.formats.style.Styler: A styled DataFrame object for pretty printing.
        """
        logger.info("📄 Fetching table data...")
        cols = "*" if not columns else ", ".join(columns)
        order_clause = f"ORDER BY {order_by}" if order_by else ""
        
        query = f"""
            SELECT {cols}
            FROM {self.full_table_path}
            {order_clause}
            LIMIT {limit}
        """
        df = self._execute_query(query)
        if df.empty:
            logger.warning("Query returned no data.")
            return None
        
        return df.style.set_properties(**{'text-align': 'left'}).set_table_styles([dict(selector='th', props=[('text-align', 'left')])])
    
    def plot_table_chart(self, dimensions: list, metrics: dict, order_by: str = None, limit: int = None):
        """
        Generates a table chart by grouping dimensions and aggregating metrics.

        Args:
            dimensions (list): A list of column names to group by.
            metrics (dict): A dictionary where keys are column names (or 'record_count') 
                            and values are the aggregation type (e.g., 'SUM', 'AVG', 'COUNT', 'COUNT_DISTINCT').
            order_by (str, optional): The column to sort by (e.g., 'total_sales DESC').
            limit (int): The number of rows to return.

        Returns:
            pandas.io.formats.style.Styler: A styled DataFrame.
            
        Example:
            metrics_to_calc = {
                'sales': 'SUM',
                'customer_id': 'COUNT_DISTINCT',
                'record_count': 'COUNT'
            }
            bq_viz.plot_table_chart(
                dimensions=['region', 'product_category'],
                metrics=metrics_to_calc
            )
        """
        if not dimensions:
            raise ValueError("You must provide at least one dimension.")
        
        logger.info("📊 Generating table chart...")
        
        # 1. Construct the SELECT clause
        select_parts = dimensions.copy()
        for metric, agg in metrics.items():
            agg_upper = agg.upper()
            if metric == 'record_count':
                # Special case for counting all rows
                select_parts.append(f"{agg_upper}(*) AS record_count")
            else:
                # Standard aggregation
                alias = f"{agg.lower()}_{metric}"
                select_parts.append(f"{agg_upper}({metric}) AS {alias}")
        
        select_clause = ",\n      ".join(select_parts)
        
        # 2. Construct the GROUP BY clause
        group_by_clause = ", ".join([str(i+1) for i in range(len(dimensions))])
        
        # 3. Construct the ORDER BY clause
        order_clause = f"ORDER BY {order_by}" if order_by else f"ORDER BY {len(select_parts)} DESC"
        limit_clause = f"LIMIT {limit}" if limit else ""

        # 4. Assemble the final query
        query = f"""
            WITH base AS (
                SELECT
                {select_clause}
                FROM {self.full_table_path}
                GROUP BY {group_by_clause}
            ),
            totals AS (
                SELECT SUM(record_count) AS total_records
                FROM base
            )
            SELECT
            b.*,
            ROUND(SAFE_DIVIDE(b.record_count, t.total_records) * 100, 2) AS record_count_pct
            FROM base b
            CROSS JOIN totals t
            {order_clause}
            {limit_clause}
        """
        
        df = self._execute_query(query)
        if df.empty:
            logger.warning("Query returned no data.")
            return None
        
        # Rename the percentage column for readability
        df = df.rename(columns={"record_count_pct": "% Records"})

        # Pretty formatting
        return df.style.format({"% Records": "{:.2f}%"}).set_properties(**{"text-align": "left"}).set_table_styles([dict(selector="th", props=[("text-align", "left")])])
    
    def plot_histogram(
        self,
        *,
        numeric_column: str,
        color_dimension: str | None = None,
        filter: str | None = None,
        limit: int | None = 100_000,
        remove_nulls: bool = True,
        bins: int | None = None,              # None → Plotly auto
        histnorm: str | None = None,          # 'percent' | 'probability' | 'density' | None
        log_x: bool = False,
        cumulative: bool = False,
        kde: bool = False,
        title: str | None = None,
    ):
        """
        Quick numeric **histogram** with optional colour split.

        Parameters
        ----------
        numeric_column : str
            Column whose distribution you want to see.
        color_dimension : str | None
            Optional categorical field to stack colour groups.
        filter : str | None
            Extra SQL WHERE condition (without leading WHERE).
        limit : int | None
            Sample row limit (defaults 100 k for performance).
        remove_nulls : bool
            Drop rows where numeric (and colour) fields are NULL.
        bins : int | None
            Number of bins; None lets Plotly choose automatically.
        histnorm : str | None
            Normalise bars: 'percent', 'probability', 'density', None (= counts).
        log_x : bool
            Log-10 scale the X-axis.
        cumulative : bool
            Show cumulative distribution.
        kde : bool
            Overlay a kernel density estimate.
        title : str | None
            Custom chart title.

        Returns
        -------
        pandas.DataFrame, plotly.graph_objs.Figure
        """
        logger.info("📊 Generating histogram…")

        where_sql = self._build_where_clause(filter)
        if remove_nulls:
            nulls = [f"{numeric_column} IS NOT NULL"]
            if color_dimension:
                nulls.append(f"{color_dimension} IS NOT NULL")
            where_sql = self._merge_where(where_sql, " AND ".join(nulls))

        bins = bins or 30
        query = f"""
            WITH edges AS (
                SELECT APPROX_QUANTILES({numeric_column}, {bins + 1}) AS qs
                FROM {self.full_table_path}
                {where_sql}
            ),
            bucketed AS (
                SELECT WIDTH_BUCKET({numeric_column}, qs) AS bucket
                       {(',' + color_dimension) if color_dimension else ''}
                FROM {self.full_table_path}, edges
                {where_sql}
            )
            SELECT
              bucket
              {(',' + color_dimension) if color_dimension else ''},
              qs[OFFSET(bucket - 1)] AS bin_start,
              qs[OFFSET(bucket)] AS bin_end,
              COUNT(*) AS n
            FROM bucketed, edges
            WHERE bucket BETWEEN 1 AND {bins}
            GROUP BY bucket{(',' + color_dimension) if color_dimension else ''}
            ORDER BY bucket
        """

        hist_df = self._execute_query(query)
        if hist_df.empty:
            logger.warning("Query returned no data.")
            return pd.DataFrame(), None

        total = hist_df['n'].sum()
        y = hist_df['n']
        if histnorm == 'percent':
            y = y / total * 100
        elif histnorm == 'probability':
            y = y / total
        elif histnorm == 'density':
            y = y / total / (hist_df['bin_end'] - hist_df['bin_start'])
        if cumulative:
            y = y.cumsum()
        hist_df['value'] = y

        fig = px.bar(hist_df, x='bin_start', y='value', color=color_dimension,
                      title=title or f"Distribution of {numeric_column}"
                      + (f" by {color_dimension}" if color_dimension else ""))
        if log_x:
            fig.update_layout(xaxis_type='log')
        ylabel = {'percent':'Percent','probability':'Probability','density':'Density'}.get(histnorm,'Count')
        fig.update_layout(margin=dict(l=0,r=0,t=40,b=0), yaxis_title=ylabel)

        if kde:
            sample_df = self.get_representative_sample(columns=[numeric_column])
            vals = sample_df[numeric_column].dropna().to_numpy()
            if log_x:
                vals = np.log10(vals[vals>0])
            if len(vals) > 1:
                kde_x = np.linspace(vals.min(), vals.max(), 200)
                density = gaussian_kde(vals)(kde_x)
                if histnorm is None:
                    bin_w = kde_x[1]-kde_x[0]
                    density = density * len(vals) * bin_w
                elif histnorm == 'percent':
                    density = density * 100
                fig.add_scatter(x=kde_x, y=density, mode='lines', name='KDE')

        fig.show()
        return hist_df, fig

    def pair_plot(
        self,
        columns: list[str],
        *,
        color_dimension: str | None = None,
        sample_rows: int = 5000,
        remove_nulls: bool = True,
    ) -> tuple[pd.DataFrame, px.scatter_matrix] | tuple[pd.DataFrame, None]:
        """Return a scatter-matrix plot for a sample of ``columns``."""

        select_cols = columns + ([color_dimension] if color_dimension else [])
        df = self.get_representative_sample(columns=select_cols)
        if df.empty:
            return pd.DataFrame(), None

        df = df[select_cols]
        if remove_nulls:
            df = df.dropna(subset=select_cols)
        if len(df) > sample_rows:
            df = df.sample(sample_rows, random_state=42)
        if df.empty:
            return pd.DataFrame(), None

        fig = px.scatter_matrix(
            df,
            dimensions=columns,
            color=color_dimension,
            height=800,
            title="Scatter Matrix",
        )
        fig.update_traces(diagonal_visible=False, showupperhalf=False)
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        fig.show()
        return df, fig

    def numeric_correlations(
        self,
        columns: list[str],
        method: str = "pearson",
    ) -> pd.DataFrame:
        """Return correlation matrix computed in BigQuery."""

        if len(columns) < 2:
            return pd.DataFrame()

        queries = []
        for i, c1 in enumerate(columns):
            for c2 in columns[i + 1 :]:
                if method.lower() == "pearson":
                    expr = f"CORR({c1}, {c2})"
                    q = f"SELECT '{c1}' AS c1, '{c2}' AS c2, {expr} AS corr FROM {self.full_table_path}"
                else:
                    q = f"""
                        SELECT '{c1}' AS c1, '{c2}' AS c2,
                               CORR(r1, r2) AS corr
                        FROM (
                            SELECT
                                RANK() OVER(ORDER BY {c1}) AS r1,
                                RANK() OVER(ORDER BY {c2}) AS r2
                            FROM {self.full_table_path}
                            WHERE {c1} IS NOT NULL AND {c2} IS NOT NULL
                        )
                    """
                queries.append(q)

        query = " UNION ALL ".join(queries)
        df = self._execute_query(query)
        if df.empty:
            return pd.DataFrame()

        mat = pd.DataFrame(np.eye(len(columns)), index=columns, columns=columns)
        for _, r in df.iterrows():
            mat.loc[r["c1"], r["c2"]] = r["corr"]
            mat.loc[r["c2"], r["c1"]] = r["corr"]
        return mat
    
    def plot_categorical_chart(
        self,
        dimensions: list,
        metrics: dict,
        *,
        orientation: str = "v",
        stacked: bool = False,
        filter: str | None = None,
        limit: int | None = None,
        order_by: str | None = None,
        show_pct: bool = True,
        bins: int | None = None,
        bin_type: str = "auto",          # "auto" | "int" | "float"
        remove_nulls: bool = True
    ):
        """
        Plot a Looker-Studio-style bar / column chart.

        Parameters
        ----------
        dimensions : list[str]
            1-N dimension columns (first one is used for X-axis).
        metrics : dict[str, str]
            Mapping of <field> → <aggregation>, e.g. {"sales": "SUM"}.
            Special metric "record_count" + "COUNT" yields COUNT(*).
        orientation : {"v", "h"}
            'v' for vertical bars, 'h' for horizontal bars.
        stacked : bool
            Plot stacked bars if multiple dimensions.
        filter : str | None
            Extra SQL in the WHERE clause (without leading "WHERE").
        limit : int | None
            LIMIT rows after aggregation (None = no limit).
        order_by : str | None
            ORDER BY clause; default = first metric DESC.
        show_pct : bool
            Add share (%) to hover.
        bins : int | None
            Number of equal-width bins for the *first* numeric dimension.
        bin_type : {"auto", "int", "float"}
            Force integer or float bin boundaries.
        remove_nulls : bool
            Drop rows where the binned dimension is NULL.

        Returns
        -------
        pandas.DataFrame, plotly.graph_objects.Figure
        """

        if not dimensions or not metrics:
            raise ValueError("Provide at least one dimension and one metric.")

        logger.info("📊 Generating categorical chart…")

        select_parts, metric_meta = [], {}
        for metric, agg in metrics.items():
            alias = f"{agg.lower()}_{metric}" if metric != "record_count" else "record_count"
            expr = "COUNT(*)" if metric == "record_count" else f"{agg.upper()}({metric})"
            select_parts.extend([*dimensions] if not select_parts else [])  # add dims once
            select_parts.append(f"{expr} AS {alias}")
            metric_meta[alias] = f"{agg.title()} {metric.replace('_', ' ').title()}"

        bin_dim, cast_dim, fmt_open, fmt_closed, cast_placeholder = None, None, None, None, None
        if bins and bins > 1:
            numeric_types = {"INT64", "INTEGER", "FLOAT", "FLOAT64", "NUMERIC", "BIGNUMERIC"}
            api_path = self.full_table_path.replace("`", "")
            for d in dimensions:
                field = next(f for f in self.client.get_table(api_path).schema if f.name == d)
                if field.field_type.upper() in numeric_types:
                    bin_dim = d
                    break
            if bin_dim:
                if bin_type == "int":
                    cast_dim = f"CAST({bin_dim} AS INT64)"
                    fmt_open, fmt_closed = "'[%d, %d)'" , "'[%d, %d]'"
                    cast_placeholder = "CAST(%s AS INT64)"
                elif bin_type == "float":
                    cast_dim = f"CAST({bin_dim} AS FLOAT64)"
                    fmt_open, fmt_closed = "'[%f, %f)'" , "'[%f, %f]'"
                    cast_placeholder = "CAST(%s AS FLOAT64)"
                else:  # auto
                    int_types = {"INT64", "INTEGER"}
                    if field.field_type.upper() in int_types:
                        cast_dim = f"CAST({bin_dim} AS INT64)"
                        fmt_open, fmt_closed = "'[%d, %d)'" , "'[%d, %d]'"
                        cast_placeholder = "CAST(%s AS INT64)"
                    else:
                        cast_dim = f"CAST({bin_dim} AS FLOAT64)"
                        fmt_open, fmt_closed = "'[%f, %f)'" , "'[%f, %f]'"
                        cast_placeholder = "CAST(%s AS FLOAT64)"
            else:
                bins = None  # nothing to bin

        user_where = self._build_where_clause(filter)     # "" or "WHERE …"
        # null_cond = f"{bin_dim} IS NOT NULL" if remove_nulls and bin_dim else ""
        extra_nulls = self._null_filter_for_dims(dimensions) if remove_nulls else ""
        merged_where = self._merge_where(user_where, extra_nulls)

        if bin_dim and bins:
            bin_cte = f"""
                WITH stats AS (
                    SELECT
                        MIN({cast_dim}) AS min_val,
                        MAX({cast_dim}) AS max_val
                    FROM {self.full_table_path}
                    {merged_where}
                ),
                binned AS (
                    SELECT
                        *,
                        CASE
                        WHEN bin_num = {bins} THEN
                            FORMAT({fmt_closed},
                                {cast_placeholder % 'min_val + (bin_num - 1) * bin_width'},
                                {cast_placeholder % 'max_val'})
                        ELSE
                            FORMAT({fmt_open},
                                {cast_placeholder % 'min_val + (bin_num - 1) * bin_width'},
                                {cast_placeholder % 'min_val + bin_num * bin_width'})
                        END AS {bin_dim}_bin
                    FROM (
                        SELECT
                            *,
                            CAST(
                                LEAST({bins},
                                    FLOOR(({cast_dim} - min_val) / bin_width) + 1)
                                AS INT64
                            ) AS bin_num
                        FROM (
                            SELECT
                                *,
                                (max_val - min_val) / {bins} AS bin_width
                            FROM stats, {self.full_table_path}
                            {merged_where}
                        )
                    )
                )
            """
            select_parts = [
                f"{bin_dim}_bin AS {bin_dim}" if col == bin_dim else col
                for col in select_parts
            ]
            query_start, source_alias = bin_cte, "binned"
        else:
            query_start, source_alias = "", self.full_table_path

        first_metric = next(iter(metric_meta))
        order_clause = (f"ORDER BY {order_by.strip() + (' DESC' if ' ' not in order_by else '')}"
                        if order_by else f"ORDER BY {first_metric} DESC")
        group_by_clause = ", ".join(f"{bin_dim}_bin" if bin_dim and col == bin_dim else col
                                    for col in dimensions)
        limit_clause = f"LIMIT {limit}" if limit else ""

        query = f"""
            {query_start}
            SELECT {', '.join(select_parts)}
            FROM {source_alias}
            {merged_where}
            GROUP BY {group_by_clause}
            {order_clause}
            {limit_clause}
        """

        df = self._execute_query(query)
        if df.empty:
            logger.warning("Query returned no data.")
            return pd.DataFrame(), None
        
        if remove_nulls:
            df = df.dropna(subset=dimensions).copy()
        else:
            for col in dimensions:
                if df[col].isna().any():
                    df[col] = df[col].astype("string").fillna("NULL")
                    
                    df[col] = pd.Categorical(df[col], ordered=True)
                    df[col] = df[col].cat.reorder_categories(
                        [c for c in df[col].cat.categories if c != "NULL"] + ["NULL"],
                        ordered=True,
                    )

        # total
        df[f"{first_metric}_pct"] = (df[first_metric] / df[first_metric].sum() * 100).round(2)

        # plot
        x_axis, y_axis = dimensions[0], first_metric
        color_axis = dimensions[1] if len(dimensions) > 1 else None
        fig = px.bar(
            df,
            x=x_axis if orientation == "v" else y_axis,
            y=y_axis if orientation == "v" else x_axis,
            color=color_axis,
            barmode="stack" if stacked else "group",
            orientation=orientation,
            title=f"{metric_meta[y_axis]} by {', '.join(dimensions)}",
            text_auto=".2s",
            height=500,
        )

        if show_pct:
            fig.update_traces(
                hovertemplate=(
                    f"<b>%{{x}}</b><br>"
                    f"{metric_meta[y_axis]}: %{{y}}<br>"
                    f"Share: %{{customdata}}%<extra></extra>"
                ),
                customdata=df[f"{first_metric}_pct"],
            )

        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        fig.show()
        return df, fig
    
    def plot_scatter_chart(
        self,
        *,
        dimension: str,
        x_metric: dict,
        y_metric: dict,
        bubble_size_metric: dict | None = None,
        color_dimension: str | None = None,
        filter: str | None = None,
        limit: int | None = None,
        remove_nulls: bool = True,
        log_x: bool = False,
        log_y: bool = False,
        trend_line: str | None = None,           # 'ols', 'lowess', None
        size_range_px: tuple[int, int] = (10, 60),
    ):
        """
        Aggregated scatter / bubble plot rendered with Plotly‑Express.

        Parameters
        ----------
        dimension : str
            Primary group‑by dimension (hover label).
        x_metric, y_metric, bubble_size_metric : dict
            {'column': str, 'aggregation': 'SUM'|'AVG'|...}.
        color_dimension : str | None
            Separate dimension for colour grouping.
        filter : str | None
            Extra SQL without leading 'WHERE'.
        limit : int | None
            LIMIT rows after aggregation (None → no limit).
        remove_nulls : bool
            Drop rows with NULL in *any* grouping dimension.
        log_x, log_y : bool
            Plot axes in log‑10 scale.
        trend_line : 'ols' | 'lowess' | None
            Add regression / LOWESS smoother.
        size_range_px : (int, int)
            Min / max bubble diameter in pixels after scaling.

        Returns
        -------
        pandas.DataFrame, plotly.graph_objs.Figure
        """
        logger.info("📈 Generating Plotly scatter plot…")

        # ───────────────────── helper to build metric parts ─────────
        def build_metric(m):
            col, agg = m["column"], m["aggregation"].upper()
            alias = f"{agg.lower()}_{col}"
            title = f"{agg.title()} {col.replace('_', ' ').title()}"
            select_expr = f"{agg}({col}) AS {alias}"
            return alias, title, select_expr

        x_alias, x_title, x_select = build_metric(x_metric)
        y_alias, y_title, y_select = build_metric(y_metric)

        selects   = [dimension]
        grp_dims  = [dimension]

        if color_dimension and color_dimension != dimension:
            selects.append(color_dimension)
            grp_dims.append(color_dimension)

        selects += [x_select, y_select]

        size_alias = size_title = size_select = None
        if bubble_size_metric:
            size_alias, size_title, size_select = build_metric(bubble_size_metric)
            selects.append(size_select)

        # ───────────────────── WHERE clause construction ────────────
        base_where = self._build_where_clause(filter)  # "" or "WHERE …"
        if remove_nulls:
            null_cond = " AND ".join(f"{d} IS NOT NULL" for d in grp_dims)
            base_where = f"{base_where + ' AND ' if base_where else 'WHERE '}{null_cond}"

        limit_clause   = f"LIMIT {limit}" if limit else ""
        group_by_clause = ", ".join(dict.fromkeys(grp_dims))  # ordered unique dims

        query = f"""
            SELECT {', '.join(selects)}
            FROM {self.full_table_path}
            {base_where}
            GROUP BY {group_by_clause}
            {limit_clause}
        """

        # ───────────────────── run & guard ──────────────────────────
        df = self._execute_query(query)
        if df.empty:
            logger.warning("Query returned no data.")
            return pd.DataFrame(), None

        # final guard: ensure requested null‑removal
        if remove_nulls:
            df = df.dropna(subset=grp_dims).copy()

        # ───────────────────── bubble size scaling ──────────────────
        if size_alias:
            low_px, hi_px = size_range_px
            vmin, vmax = df[size_alias].min(), df[size_alias].max()
            if vmin == vmax:
                df["_size_px"] = (low_px + hi_px) / 2
            else:
                df["_size_px"] = (df[size_alias] - vmin) / (vmax - vmin) * (hi_px - low_px) + low_px
        else:
            df["_size_px"] = (sum(size_range_px) / 2)

        # ───────────────────── Plotly scatter ───────────────────────
        fig = px.scatter(
            df,
            x=x_alias,
            y=y_alias,
            size="_size_px",
            color=(color_dimension if color_dimension else dimension),
            hover_name=dimension,
            size_max=max(size_range_px),
            log_x=log_x,
            log_y=log_y,
            trendline=trend_line,
            trendline_color_override="black",
            color_discrete_sequence=px.colors.qualitative.Bold,
            title=f"{y_title} vs. {x_title}",
            height=650,
        )

        # nicer axis titles
        fig.update_layout(
            xaxis_title=x_title,
            yaxis_title=y_title,
            margin=dict(l=0, r=0, t=40, b=0),
            legend_title_text=color_dimension or dimension,
        )

        fig.show()
        return df, fig

    def plot_sunburst(self, dimensions: list, filter: str = None):
        """
        Generates an interactive sunburst chart showing record counts across a hierarchy.

        Args:
            dimensions (list): An ordered list of categorical columns for the hierarchy.
            filter (str, optional): An SQL filter condition.
        """
        logger.info(
            "☀️ Generating sunburst chart for hierarchy: %s",
            " -> ".join(dimensions),
        )
        where_clause = self._build_where_clause(filter)
        dims_str = ", ".join(dimensions)
        
        query = f"""
            SELECT
              {dims_str},
              COUNT(*) AS record_count
            FROM {self.full_table_path}
            {where_clause}
            GROUP BY {dims_str}
        """
        df = self._execute_query(query)
        if df.empty:
            logger.warning("Query returned no data.")
            return pd.DataFrame(), None
        
        fig = px.sunburst(
            df, 
            path=dimensions, 
            values='record_count',
            title=f"Record Count by {' and '.join(dimensions)}"
        )
        fig.update_traces(textinfo="label+percent parent")
        fig.update_layout(
            autosize=True,
            height=800,
            margin=dict(t=60, l=0, r=0, b=0)
        )
        fig.show()

        return df, fig

    def plot_boxen_chart(
        self,
        *,
        numeric_column: str,
        category_dimension: str | None = None,
        filter: str | None = None,
        limit: int | None = 100_000,
        remove_nulls: bool = True,
        log_scale: bool = False,
        palette: str = "Set3",
    ):
        """
        Draw a seaborn **boxen plot** of a numeric column, optionally split by a
        categorical dimension.

        Parameters
        ----------
        numeric_column : str
            The numeric field to visualise.
        category_dimension : str | None
            When provided, one boxen per category.
        filter : str | None
            Extra SQL (without leading WHERE).
        limit : int | None
            LIMIT rows passed to the plot. Set smaller for very wide tables.
        remove_nulls : bool
            Drop rows where the numeric or category field is NULL.
        log_scale : bool
            Apply log10 transform to the numeric values before plotting.
        palette : str
            Seaborn palette name.

        Returns
        -------
        pandas.DataFrame, matplotlib.axes.Axes
        """
        logger.info("📊 Generating boxen plot…")

        # ---------- 1. build query -----------------------------------
        select_cols = [numeric_column]
        if category_dimension:
            select_cols.insert(0, category_dimension)

        base_where = self._build_where_clause(filter)

        if remove_nulls:
            null_conds = [f"{numeric_column} IS NOT NULL"]
            if category_dimension:
                null_conds.append(f"{category_dimension} IS NOT NULL")
            null_where = " AND ".join(null_conds)
            base_where = f"{base_where + ' AND ' if base_where else 'WHERE '}{null_where}"

        limit_clause = f"LIMIT {limit}" if limit else ""

        query = f"""
            SELECT {', '.join(select_cols)}
            FROM {self.full_table_path}
            {base_where}
            {limit_clause}
        """

        # ---------- 2. run -------------------------------------------
        df = self._execute_query(query)
        if df.empty:
            logger.warning("Query returned no data.")
            return pd.DataFrame(), None

        # ---------- 3. optional log transform ------------------------
        plot_col = numeric_column
        if log_scale:
            # add small offset if zeros present
            if (df[numeric_column] <= 0).any():
                offset = df[numeric_column][df[numeric_column] > 0].min() / 10
                df["_log_value"] = np.log10(df[numeric_column] + offset)
            else:
                df["_log_value"] = np.log10(df[numeric_column])
            plot_col = "_log_value"

        # ---------- 4. plot ------------------------------------------
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(10, 6))

        sns.boxenplot(
            data=df,
            x=category_dimension if category_dimension else None,
            y=plot_col,
            palette=palette,
            ax=ax,
        )

        # titles & labels
        y_label = f"{numeric_column} (log10)" if log_scale else numeric_column
        if category_dimension:
            ax.set_xlabel(category_dimension)
        else:
            ax.set_xlabel("")  # no category axis
        ax.set_ylabel(y_label)
        ax.set_title(f"Distribution of {numeric_column}"
                    + (f" by {category_dimension}" if category_dimension else ""))

        plt.tight_layout()
        plt.show()

        return df, ax
    
    def plot_violin_chart(
        self,
        *,
        numeric_column: str,
        category_dimension: str | None = None,
        filter: str | None = None,
        limit: int | None = 100_000,
        remove_nulls: bool = True,
        log_scale: bool = False,
        palette: str = "Set2",
        split: bool = False,           # for two‑class violins
    ):
        """
        Seaborn violin plot (distribution) of a numeric column.

        Parameters
        ----------
        numeric_column : str
            The numeric field whose distribution you want to visualise.
        category_dimension : str | None
            If provided, draw one violin per category.
        filter : str | None
            Extra SQL without leading 'WHERE'.
        limit : int | None
            LIMIT rows fetched. 100 k is plenty for KDE.
        remove_nulls : bool
            Drop rows where the numeric (and, if supplied, category) field is NULL.
        log_scale : bool
            Apply log10 transform to the numeric values before plotting.
        palette : str
            Seaborn palette name.
        split : bool
            If `category_dimension` has exactly two levels, set True to get the
            classic split‑violin view.

        Returns
        -------
        pandas.DataFrame, matplotlib.axes.Axes
        """
        logger.info("🎻 Generating violin plot…")

        # ---------- 1. build query -----------------------------------
        select_cols = [numeric_column]
        if category_dimension:
            select_cols.insert(0, category_dimension)

        where_sql = self._build_where_clause(filter)
        if remove_nulls:
            null_conds = [f"{numeric_column} IS NOT NULL"]
            if category_dimension:
                null_conds.append(f"{category_dimension} IS NOT NULL")
            where_sql = f"{where_sql + ' AND ' if where_sql else 'WHERE '}{' AND '.join(null_conds)}"

        query = f"""
            SELECT {', '.join(select_cols)}
            FROM {self.full_table_path}
            {where_sql}
            {f'LIMIT {limit}' if limit else ''}
        """

        df = self._execute_query(query)
        if df.empty:
            logger.warning("Query returned no data.")
            return pd.DataFrame(), None

        # ---------- 2. optional log transform ------------------------
        plot_col = numeric_column
        if log_scale:
            if (df[numeric_column] <= 0).any():
                offset = df[numeric_column][df[numeric_column] > 0].min() / 10
                df["_log_value"] = np.log10(df[numeric_column] + offset)
            else:
                df["_log_value"] = np.log10(df[numeric_column])
            plot_col = "_log_value"

        # ---------- 3. plot ------------------------------------------
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(10, 6))

        sns.violinplot(
            data=df,
            x=category_dimension if category_dimension else None,
            y=plot_col,
            palette=palette,
            split=split if category_dimension else False,
            inner="quartile",
            ax=ax,
        )

        y_label = f"{numeric_column} (log10)" if log_scale else numeric_column
        ax.set_ylabel(y_label)
        if category_dimension:
            ax.set_xlabel(category_dimension)
        else:
            ax.set_xlabel("")
        title = f"Distribution of {numeric_column}"
        if category_dimension:
            title += f" by {category_dimension}"
        ax.set_title(title)

        plt.tight_layout()
        plt.show()

        return df, ax
    
    def plot_pie_chart(
        self,
        *,
        dimension: str,
        metric: dict | None = None,        # None → COUNT(*)
        filter: str | None = None,
        limit: int | None = 10,            # show top-N slices
        remove_nulls: bool = True,
        hole: float = 0.0,                 # 0 = pie, 0.3-0.6 nice donut
        title: str | None = None,
    ):
        """
        Pie or donut chart (set *hole*) of a single categorical dimension.

        Parameters
        ----------
        dimension : str
            The categorical column to slice.
        metric : dict | None
            {'column': str, 'aggregation': 'SUM'|'AVG'|...}.
            If None, slice sizes are raw record counts.
        filter : str | None
            SQL WHERE clause fragment.
        limit : int | None
            Keep top-N categories (slice count); rest rolled into "Other".
        remove_nulls : bool
            Exclude NULL rows from the dimension.
        hole : float
            0.0 → classic pie; 0.25…0.6 → donut; 0.9 → thin ring.
        title : str | None
            Custom chart title.

        Returns
        -------
        pandas.DataFrame, plotly.graph_objs.Figure
        """
        logger.info("🥧 Generating pie / donut chart…")

        # ───── 1. build SELECT parts ──────────────────────────────
        if metric:
            col, agg = metric["column"], metric["aggregation"].upper()
            value_alias = f"{agg.lower()}_{col}"
            select_part = f"{agg}({col}) AS {value_alias}"
        else:
            value_alias = "record_count"
            select_part = "COUNT(*) AS record_count"

        base_where = self._build_where_clause(filter)
        if remove_nulls:
            null_cond = f"{dimension} IS NOT NULL"
            base_where = f"{base_where + ' AND ' if base_where else 'WHERE '}{null_cond}"

        limit_clause = f"LIMIT {limit}" if limit else ""

        query = f"""
            WITH ranked AS (
                SELECT {dimension},
                    {select_part}
                FROM {self.full_table_path}
                {base_where}
                GROUP BY {dimension}
                ORDER BY {value_alias} DESC
                {limit_clause}
            )
            SELECT * FROM ranked
        """

        df = self._execute_query(query)
        if df.empty:
            logger.warning("Query returned no data.")
            return pd.DataFrame(), None

        # ───── 2. Plotly pie ──────────────────────────────────────
        fig = px.pie(
            df,
            names=dimension,
            values=value_alias,
            hole=hole,                                      # 0 → pie, >0 donut
            title=title or f"{value_alias.replace('_', ' ').title()} by {dimension}",
            color_discrete_sequence=px.colors.qualitative.Set3,
        )

        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        fig.show()

        return df, fig
    
    def analyze_numeric_column(self, numeric_column: str):
        """
        Performs a full descriptive analysis of a numeric column using BigQuery aggregations.

        Args:
            numeric_column (str): The numeric column to analyze.
        
        Returns:
            pd.DataFrame: A DataFrame containing the descriptive statistics.
        """
        logger.info("🔍 Analyzing numeric column: %s", numeric_column)
        query = f"""
            SELECT
                COUNT(*) AS total_rows,
                COUNTIF({numeric_column} IS NULL) AS null_count,
                COUNT({numeric_column}) AS non_null_count,
                AVG({numeric_column}) AS mean,
                STDDEV({numeric_column}) AS std_dev,
                VAR_SAMP({numeric_column}) AS variance,
                MIN({numeric_column}) AS min,
                MAX({numeric_column}) AS max,
                SKEWNESS({numeric_column}) AS skewness,
                KURTOSIS({numeric_column}) AS kurtosis,
                APPROX_QUANTILES({numeric_column}, 4) AS quartiles
            FROM {self.full_table_path}
        """
        df = self._execute_query(query)
        if df.empty or df.iloc[0]['total_rows'] == 0:
            logger.warning("Could not analyze column.")
            return {}

        stats = df.iloc[0].to_dict()
        
        # Post-process for readability
        quartiles = stats.pop('quartiles')
        summary = {
            # "Total Rows": int(stats['total_rows']),
            "Null Count": int(stats['null_count']),
            "Null %": (stats['null_count'] / stats['total_rows']) * 100,
            "Mean": stats['mean'],
            "Std Dev": stats['std_dev'],
            "Variance": stats['variance'],
            "Min": stats['min'],
            "Skewness": stats.get('skewness'),
            "Kurtosis": stats.get('kurtosis'),
            "25% (Q1)": quartiles[1],
            "50% (Median)": quartiles[2], # 50th percentile
            "75% (Q3)": quartiles[3],
            "Max": stats['max'],
            "IQR": quartiles[3] - quartiles[1]
        }
        
        return summary

    def analyze_categorical_column(self, categorical_column: str, top_n_values: int = 10):
        """
        Performs a full descriptive analysis of a categorical column using BigQuery aggregations.

        Args:
            categorical_column (str): The categorical column to analyze.
            top_n_values (int): The number of most frequent values to display.
        
        Returns:
            pd.DataFrame: A DataFrame containing the descriptive statistics.
        """
        logger.info("🔍 Analyzing categorical column: %s", categorical_column)
        query = f"""
            WITH base_stats AS (
                SELECT
                    COUNT(*) AS total_rows,
                    COUNTIF({categorical_column} IS NULL) AS null_count,
                    COUNT(DISTINCT {categorical_column}) AS unique_count
                FROM {self.full_table_path}
            ),
            top_values AS (
                SELECT
                    {categorical_column} as value,
                    COUNT(*) as count
                FROM {self.full_table_path}
                WHERE {categorical_column} IS NOT NULL
                GROUP BY 1
                ORDER BY 2 DESC
                LIMIT {top_n_values}
            )
            SELECT
                (SELECT total_rows FROM base_stats) as total_rows,
                (SELECT null_count FROM base_stats) as null_count,
                (SELECT unique_count FROM base_stats) as unique_count,
                ARRAY_AGG(STRUCT(tv.value, tv.count)) AS top_n
            FROM top_values tv
        """
        df = self._execute_query(query)
        if df.empty or df.iloc[0]['total_rows'] is None:
            logger.warning("Could not analyze column.")
            return {}

        stats = df.iloc[0].to_dict()

        summary = {
            # "Total Rows": int(stats['total_rows']),
            "Null Count": int(stats['null_count']),
            "Null %": (stats['null_count'] / stats['total_rows']) * 100,
            "Unique Values": int(stats['unique_count']),
            "Top Values": stats['top_n']
        }
        
        return summary
    
    def analyze_all_numeric(self, numeric_columns: list = None):
        """
        Runs descriptive analysis for all numeric columns and returns a single summary DataFrame.
        """
        logger.info("🤖 Starting automated analysis for all numeric columns...")
        
        all_results = []
        numeric_columns = [col for col in numeric_columns if col in self.numeric_columns] if numeric_columns else self.numeric_columns
        for col in numeric_columns:
            summary = self.analyze_numeric_column(col)
            if summary:
                summary['column_name'] = col
                all_results.append(summary)
        
        if not all_results:
            logger.warning("No numeric data to analyze.")
            return pd.DataFrame()
            
        # Convert list of dicts to a DataFrame
        summary_df = pd.DataFrame(all_results).set_index('column_name')
        
        # Define a nice column order
        col_order = [
            'Null Count', 'Null %', 'Mean', 'Std Dev', 'Variance', 'Skewness',
            'Kurtosis', 'Min', '25% (Q1)', '50% (Median)', '75% (Q3)', 'Max',
            'IQR'
        ]
        summary_df = summary_df[col_order]

        logger.info("✅ Analysis complete.")
        # Return a styled DataFrame for pretty printing in notebooks
        return summary_df.style.format({
            'Null Count': '{:,.2f}',
            'Null %': '{:.2f}%',
            'Mean': '{:,.2f}',
            'Std Dev': '{:,.2f}',
            'Variance': '{:,.2f}',
            'Skewness': '{:,.2f}',
            'Kurtosis': '{:,.2f}',
            'Min': '{:,.2f}',
            '25% (Q1)': '{:,.2f}',
            '50% (Median)': '{:,.2f}',
            '75% (Q3)': '{:,.2f}',
            'Max': '{:,.2f}',
            'IQR': '{:,.2f}'
        }).background_gradient(cmap='viridis', subset=['Null %'])

    def analyze_all_categorical(self, top_n_values: int = 5, categorical_columns: list = None):
        """
        Runs analysis for all categorical columns and returns a single summary DataFrame
        with top-N values expanded into separate columns as percentages.
        """
        logger.info("🤖 Starting automated analysis for all categorical columns...")

        all_results = []
        categorical_columns = [col for col in categorical_columns if col in self.categorical_columns] if categorical_columns else self.categorical_columns
        for col in categorical_columns:
            summary = self.analyze_categorical_column(col, top_n_values=top_n_values)
            if summary:
                # Build the base record
                row = {
                    "column_name": col,
                    "Null %": summary["Null %"],
                    "Unique Values": summary["Unique Values"],
                }

                # Mode (top value)
                top_vals = summary.get("Top Values", [])
                if len(top_vals):
                    row["Mode"] = top_vals[0]["value"]
                    row["Mode Freq."] = (top_vals[0]["count"] / (summary["Null Count"] + sum(v["count"] for v in top_vals))) * 100
                else:
                    row["Mode"] = None
                    row["Mode Freq."] = 0.0

                # Expand Top 1 … Top N as percentages
                total_rows = summary["Null Count"] + sum(v["count"] for v in top_vals)
                for rank in range(1, top_n_values + 1):
                    if rank - 1 < len(top_vals):
                        value = top_vals[rank - 1]["value"]
                        pct = (top_vals[rank - 1]["count"] / total_rows) * 100
                    else:
                        value, pct = None, None
                    row[f"Top {rank}"] = f"{value} ({pct:.2f}%)" if pct is not None else None

                all_results.append(row)

        if not all_results:
            logger.warning("No categorical data to analyze.")
            return pd.DataFrame()
        
        summary_df = pd.DataFrame(all_results).set_index('column_name')
        
        # # Define a nice column order
        # col_order = [
        #     'Null %', 'Unique Values', 'Mode', 'Mode Freq.', 'Top Values'
        # ]
        # summary_df = summary_df[col_order]
        
        logger.info("✅ Analysis complete.")
        return summary_df.style.background_gradient(cmap='Reds', subset=['Null %'])

    # ------------------------------------------------------------------
    # Sampling & bias evaluation utilities
    # ------------------------------------------------------------------
    def fetch_sample(self, n: int, *, where: str | None = None) -> pd.DataFrame:
        """Return a random sample of ``n`` rows from the table."""
        query = (
            f"SELECT * FROM {self.full_table_path} "
            f"{self._build_where_clause(where)} ORDER BY RAND() LIMIT {n}"
        )
        return self._execute_query(query)

    def evaluate_sample_bias(
        self,
        *,
        sample_rows: int = 1000,
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """Compare a random sample against the full table.

        Numeric columns are tested with the Kolmogorov–Smirnov test and
        categorical columns with a Chi-squared test. ``alpha`` controls the
        significance threshold for the ``biased`` flag.
        """

        sample_df = self.fetch_sample(sample_rows)
        results: list[dict] = []

        # numeric columns – KS test
        for col in self.numeric_columns:
            ref = self.fetch_sample(sample_rows, where=f"{col} IS NOT NULL")[col]
            if ref.empty or sample_df[col].dropna().empty:
                continue
            stat, pval = ks_2samp(sample_df[col].dropna(), ref.dropna())
            results.append({
                "column": col,
                "test": "ks",
                "statistic": stat,
                "p_value": pval,
                "biased": pval < alpha,
            })

        # categorical columns – Chi^2
        for col in self.categorical_columns:
            full_counts = self._execute_query(
                f"SELECT {col}, COUNT(*) as n FROM {self.full_table_path} "
                f"WHERE {col} IS NOT NULL GROUP BY {col}"
            )
            if full_counts.empty:
                continue
            sample_counts = sample_df[col].value_counts().rename("sample")
            merged = full_counts.set_index(col)["n"].rename("population")
            both = (
                pd.concat([sample_counts, merged], axis=1)
                .fillna(0)
                .astype(int)
            )
            chi2, pval, _, _ = chi2_contingency(both.T.values)
            results.append({
                "column": col,
                "test": "chi2",
                "statistic": chi2,
                "p_value": pval,
                "biased": pval < alpha,
            })

        return pd.DataFrame(results)

    def get_representative_sample(
        self,
        columns: list[str] | None = None,
        max_bytes: int | None = None,
        refresh: bool = False,
    ) -> pd.DataFrame:
        """Return and cache a representative sample of the table."""

        cols = columns or self.columns
        key = ",".join(sorted(cols))
        fname = self.sample_cache_dir / f"rep_{hashlib.md5(key.encode()).hexdigest()}.csv"

        # We intentionally do not load previously cached samples from disk
        # so queries execute at least once per session. The sampled data is still
        # written to ``fname`` for offline inspection.

        if (
            self.rep_sample_df is not None
            and self.rep_sample_columns_key == key
            and not refresh
        ):
            return self.rep_sample_df.copy()

        limit_bytes = max_bytes or self.max_result_bytes
        size_map = {
            "numeric": 8,
            "string": 20,
            "boolean": 1,
            "datetime": 8,
            "complex": 50,
            "geographic": 16,
            "other": 8,
        }
        cat_lookup = dict(zip(self.schema_df["column_name"], self.schema_df["category"]))
        row_bytes = sum(size_map.get(cat_lookup.get(c, "other"), 8) for c in cols)
        row_bytes = max(1, row_bytes)
        max_rows = max(1, limit_bytes // row_bytes)

        query = (
            f"SELECT {', '.join(cols)} FROM {self.full_table_path} TABLESAMPLE SYSTEM (1 PERCENT) "
            f"LIMIT {max_rows}"
        )
        df = self._execute_query(query)
        if not df.empty:
            self.evaluate_sample_bias(sample_rows=min(1000, len(df)))
        self.rep_sample_df = df
        self.rep_sample_columns_key = key
        try:
            df.to_csv(fname, index=False)
        except Exception as e:
            logger.warning("⚠️ Could not cache sample to %s: %s", fname, e)
        return df.copy()

    def missingness_correlation(
        self,
        columns: list[str] | None = None,
        sample_rows: int = 100_000,
    ) -> pd.DataFrame:
        """Return the correlation matrix of null indicators for ``columns``."""

        cols = columns or self.columns
        df = self.get_representative_sample(columns=cols)
        if df.empty:
            return pd.DataFrame()
        df = df[cols]
        if len(df) > sample_rows:
            df = df.sample(sample_rows, random_state=42)
        miss = df[cols].isna().astype(int)
        return miss.corr()

    def frequent_missing_patterns(
        self,
        columns: list[str] | None = None,
        *,
        top_n: int = 10,
        sample_rows: int = 100_000,
    ) -> pd.DataFrame:
        """Identify the most common combinations of missing values."""

        cols = columns or self.columns
        df = self.get_representative_sample(columns=cols)
        if df.empty:
            return pd.DataFrame()
        df = df[cols]
        if len(df) > sample_rows:
            df = df.sample(sample_rows, random_state=42)

        mask = df[cols].isna()
        combo = mask.apply(lambda r: '|'.join(r.index[r]), axis=1)
        counts = combo.value_counts().reset_index()
        counts.columns = ["missing_combination", "count"]
        counts["pct"] = counts["count"] / len(combo) * 100
        return counts.head(top_n)

    def missingness_map(
        self,
        columns: list[str] | None = None,
        *,
        sample_rows: int = 100_000,
    ) -> tuple[pd.DataFrame, plt.Axes | None, pd.DataFrame]:
        """Visualise missingness and run simple MCAR/MAR tests.

        Returns a tuple of ``(mask_df, fig, mcar_results)`` where ``mask_df``
        is the boolean missingness DataFrame, ``fig`` a seaborn heatmap, and
        ``mcar_results`` contains per-column MCAR/MAR classifications based on
        pairwise statistical tests against all other columns.
        """

        cols = columns or self.columns
        df = self.get_representative_sample(columns=cols)
        if df.empty:
            return pd.DataFrame(), None, pd.DataFrame()
        df = df[cols]
        if len(df) > sample_rows:
            df = df.sample(sample_rows, random_state=42)

        mask = df[cols].isna()

        # ---- 1. heatmap ----
        sns.set_theme(style="white")
        fig, ax = plt.subplots(figsize=(max(6, len(cols) * 0.5), 4))
        sns.heatmap(mask.T, cbar=False, ax=ax)
        ax.set_ylabel("column")
        ax.set_xlabel("row")
        ax.set_title("Missingness Map")

        # ---- 2. simple MCAR/MAR tests ----
        results = []
        for col in cols:
            m = mask[col]
            pvals = []
            for other in [c for c in cols if c != col]:
                if other in self.numeric_columns:
                    stat, p = ks_2samp(
                        df.loc[m, other].dropna(),
                        df.loc[~m, other].dropna(),
                    )
                    pvals.append(p)
                else:
                    ct = pd.crosstab(m, df[other])
                    if ct.size <= 0:
                        continue
                    chi2, p, *_ = chi2_contingency(ct)
                    pvals.append(p)
            mcar = all(p > 0.05 for p in pvals) if pvals else True
            results.append({"column": col, "MCAR": mcar})

        res_df = pd.DataFrame(results)
        return mask, ax, res_df

    def generate_splits(
        self,
        *,
        target_column: str,
        method: str = "random",
        val_size: float = 0.2,
        test_size: float = 0.1,
        time_column: str | None = None,
        random_state: int = 42,
    ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
        """Create train/validation/test splits and return class balance stats."""

        df = self._execute_query(f"SELECT * FROM {self.full_table_path}")
        if df.empty:
            return {}, pd.DataFrame()

        if method == "time" and time_column:
            df = df.sort_values(time_column)
            n = len(df)
            test_n = int(n * test_size)
            val_n = int(n * val_size)
            test_df = df.iloc[-test_n:]
            val_df = df.iloc[-test_n - val_n:-test_n]
            train_df = df.iloc[: -test_n - val_n]
        else:
            strat = df[target_column] if method == "stratified" else None
            train_df, temp_df = train_test_split(
                df,
                test_size=val_size + test_size,
                stratify=strat,
                random_state=random_state,
            )
            strat_temp = temp_df[target_column] if method == "stratified" else None
            val_rel = val_size / (val_size + test_size)
            val_df, test_df = train_test_split(
                temp_df,
                test_size=1 - val_rel,
                stratify=strat_temp,
                random_state=random_state,
            )

        splits = {"train": train_df, "validation": val_df, "test": test_df}

        balance = {}
        for name, frame in splits.items():
            counts = frame[target_column].value_counts(normalize=True)
            balance[name] = counts
        balance_df = pd.DataFrame(balance).fillna(0) * 100
        balance_df.index.name = target_column

        return splits, balance_df

    # ------------------------------------------------------------------
    #  Advanced analytics helpers
    # ------------------------------------------------------------------
    def project_2d(
        self,
        *,
        method: str,
        columns: list[str],
        sample_rows: int = 10000,
    ) -> tuple[pd.DataFrame, px.scatter] | tuple[pd.DataFrame, None]:
        """Return a 2-D projection of ``columns`` using BigQuery ML."""

        algo = method.lower()
        tvf = {
            "pca": "ML.PCA",
            "tsne": "ML.TSNE",
            "umap": "ML.UMAP",
        }.get(algo)
        if tvf is None:
            raise ValueError("method must be 'pca', 'tsne', or 'umap'")

        col_sql = ", ".join(columns)
        not_null = " AND ".join(f"{c} IS NOT NULL" for c in columns)
        query = f"""
            SELECT
              principal_component_1 AS dim1,
              principal_component_2 AS dim2
            FROM {tvf}(
              (SELECT {col_sql} FROM {self.full_table_path} WHERE {not_null}),
              STRUCT(2 AS num_principal_components)
            )
        """

        df = self._execute_query(query)
        if df.empty:
            return pd.DataFrame(), None

        fig = px.scatter(df, x="dim1", y="dim2", opacity=0.4,
                         title=f"{method.upper()} projection (2D)")
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        fig.show()
        return df, fig

    def partial_dependence(
        self,
        *,
        feature: str,
        target: str,
        bins: int = 10,
    ) -> tuple[pd.DataFrame, px.line] | tuple[pd.DataFrame, None]:
        """Compute simple 1-D partial dependence via BigQuery aggregation."""

        query = f"""
            WITH stats AS (
                SELECT MIN({feature}) AS min_val, MAX({feature}) AS max_val
                FROM {self.full_table_path}
            )
            SELECT
              CAST(({feature} - stats.min_val) /
                   NULLIF(stats.max_val - stats.min_val,0) * {bins} AS INT64) AS bin_id,
              AVG({target}) AS avg_target,
              COUNT(*) AS n
            FROM {self.full_table_path}, stats
            WHERE {feature} IS NOT NULL AND {target} IS NOT NULL
            GROUP BY bin_id
            ORDER BY bin_id
        """
        df = self._execute_query(query)
        if df.empty:
            return pd.DataFrame(), None

        fig = px.line(df, x="bin_id", y="avg_target",
                      markers=True, title=f"Partial dependence of {target} on {feature}")
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        fig.show()
        return df, fig

    @staticmethod
    def hopkins_statistic(X: np.ndarray, n_samples: int | None = None) -> float:
        """Compute the Hopkins statistic for cluster tendency."""

        n, _ = X.shape
        m = n_samples or min(100, n - 1)
        rng = np.random.default_rng(42)
        idx = rng.choice(n, m, replace=False)
        sample = X[idx]

        uniform = np.empty_like(sample)
        for j in range(X.shape[1]):
            mn, mx = X[:, j].min(), X[:, j].max()
            uniform[:, j] = rng.uniform(mn, mx, size=m)

        nbrs = NearestNeighbors(n_neighbors=1).fit(X)
        u_dist, _ = nbrs.kneighbors(uniform)
        w_dist, _ = nbrs.kneighbors(sample)
        H = u_dist.sum() / (u_dist.sum() + w_dist.sum())
        return float(H)

    @staticmethod
    def silhouette_score_estimate(X: np.ndarray, n_clusters: int = 2) -> float:
        """Compute a rough silhouette score using KMeans."""

        if len(X) <= n_clusters:
            return float("nan")
        labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit_predict(X)
        return float(silhouette_score(X, labels))
