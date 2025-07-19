import logging
from google.cloud import bigquery

logger = logging.getLogger(__name__)

def execute_query_with_guard(client, sql, job_config=None, max_gb_processed=1.0, max_result_bytes=2_000_000_000):
    """Execute a BigQuery SQL query with a dry-run and result-size guard."""
    cfg = bigquery.QueryJobConfig(dry_run=True, use_query_cache=True)
    dry_job = client.query(sql, job_config=cfg)
    if dry_job.total_bytes_processed > max_gb_processed * 1e9:
        raise RuntimeError(
            f"Query would process {dry_job.total_bytes_processed/1e9:.2f} GB "
            f"(limit {max_gb_processed:.2f} GB). Aborting."
        )

    est_bytes = 0
    try:
        plan_job = client.query(f"EXPLAIN {sql}")
        plan_job.result()
        if plan_job.query_plan:
            final = plan_job.query_plan[-1]
            est_bytes = int(
                final._properties.get("statistics", {}).get("estimatedBytes", 0)
            )
    except Exception:
        est_bytes = 0

    if est_bytes and est_bytes > max_result_bytes:
        raise RuntimeError(
            f"Query would return {est_bytes/1e9:.2f} GB "
            f"(limit {max_result_bytes/1e9:.2f} GB). Aborting."
        )

    logger.info("ℹ️ Query will process %0.2f GB", dry_job.total_bytes_processed / 1e9)

    job = client.query(sql, job_config=job_config)
    return job.to_dataframe()
