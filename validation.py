"""Utilities for generating Great Expectations suites from EDA results."""
from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from great_expectations.core import ExpectationSuite
from great_expectations.expectations.expectation_configuration import (
    ExpectationConfiguration,
)

if TYPE_CHECKING:  # pragma: no cover
    from .bigquery_visualizer import BigQueryVisualizer
    from .analysis_context import AnalysisContext


def build_expectation_suite(viz: 'BigQueryVisualizer', ctx: 'AnalysisContext') -> ExpectationSuite:
    """Create a basic :class:`ExpectationSuite` from profiling results."""
    schema_df: pd.DataFrame | None = ctx.get_table('profiling.schema_overview')
    if schema_df is None and hasattr(viz, 'schema_df'):
        schema_df = viz.schema_df
    if schema_df is None:
        raise ValueError('Schema information not found in context.')

    missing_df: pd.DataFrame | None = ctx.get_table('quality.missing_pct')
    uniq_df: pd.DataFrame | None = ctx.get_table('quality.unique_ratio')

    suite = ExpectationSuite('generated_suite')

    # Map BigQuery types to GE-friendly types (pandas execution engine)
    type_map = {
        'INT64': 'int64', 'INTEGER': 'int64', 'NUMERIC': 'float64', 'BIGNUMERIC': 'float64',
        'FLOAT64': 'float64', 'BOOL': 'bool', 'BOOLEAN': 'bool', 'STRING': 'str', 'BYTES': 'bytes',
        'DATE': 'datetime64[ns]', 'DATETIME': 'datetime64[ns]', 'TIMESTAMP': 'datetime64[ns]', 'TIME': 'datetime64[ns]',
        'GEOGRAPHY': 'object', 'ARRAY': 'object', 'STRUCT': 'object',
    }

    for _, row in schema_df.iterrows():
        col = row['column_name']
        dtype = type_map.get(str(row['data_type']).upper(), 'object')
        suite.add_expectation_configuration(
            ExpectationConfiguration(
                "expect_column_values_to_be_of_type",
                {"column": col, "type_": dtype},
            )
        )

        if missing_df is not None:
            mrow = missing_df[missing_df['column'] == col]
            if not mrow.empty and float(mrow['missing_pct'].iloc[0]) == 0:
                suite.add_expectation_configuration(
                    ExpectationConfiguration(
                        "expect_column_values_to_not_be_null",
                        {"column": col},
                    )
                )

        if uniq_df is not None:
            urow = uniq_df[uniq_df['column'] == col]
            if not urow.empty and float(urow['unique_ratio'].iloc[0]) == 100.0:
                suite.add_expectation_configuration(
                    ExpectationConfiguration(
                        "expect_column_values_to_be_unique",
                        {"column": col},
                    )
                )

    return suite
