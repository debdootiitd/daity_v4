"""BigQuery client wrapper.

Phase 0 only needs lightweight metadata + sample queries. The Storage Read /
Arrow path is added in Phase 1 alongside the Parquet cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from google.api_core import exceptions as gcp_exceptions
from google.cloud import bigquery

from daity.utils.env import BQConfig
from daity.utils.logging import get_logger

log = get_logger(__name__)

# Hard ceiling per query so a malformed audit can't run up a billing surprise.
# 10 GiB is plenty for Phase 0 metadata + sample queries; tighten in CI if desired.
DEFAULT_BYTES_BUDGET: int = 10 * 2**30


@dataclass(frozen=True, slots=True)
class TableMeta:
    """Lightweight table metadata returned by `BQClient.table_info`."""

    table_id: str
    full_table_id: str
    num_rows: int
    num_bytes: int
    created: str | None
    modified: str | None
    schema: list[dict[str, Any]]
    partitioning: dict[str, Any] | None
    clustering_fields: list[str] | None


class BQClient:
    """Thin wrapper around `google.cloud.bigquery.Client`.

    On construction, auto-detects the dataset's actual location and (if it
    differs from the configured one) replaces the client with a correctly-
    located one. This avoids the silent 404 trap when an Indian-data dataset
    lives in `asia-south1` but the env defaulted to `US`.
    """

    def __init__(self, cfg: BQConfig, *, bytes_budget: int = DEFAULT_BYTES_BUDGET) -> None:
        self.cfg = cfg
        self.bytes_budget = bytes_budget

        # Initial client (may be at the wrong location).
        client = bigquery.Client(project=cfg.project, location=cfg.location)

        # Auto-detect dataset location.
        ds_ref = client.dataset(cfg.dataset)
        try:
            ds = client.get_dataset(ds_ref)
            actual_loc = ds.location
        except gcp_exceptions.NotFound as exc:
            msg = (
                f"Dataset {cfg.fq_dataset} not found at location={cfg.location!r}. "
                f"If the dataset lives in a different region, leave DAITY_BQ_LOCATION "
                f"unset and let auto-detection handle it. Underlying error: {exc}"
            )
            raise RuntimeError(msg) from exc

        if cfg.location and cfg.location.lower() != actual_loc.lower():
            log.warning(
                "DAITY_BQ_LOCATION=%s but dataset is in %s; rebinding client to dataset's location",
                cfg.location,
                actual_loc,
            )
            client = bigquery.Client(project=cfg.project, location=actual_loc)
        self._client = client
        self.location = actual_loc

        log.info(
            "BQClient ready: project=%s dataset=%s location=%s",
            cfg.project,
            cfg.dataset,
            self.location,
        )

    # ----- Introspection -----

    def list_tables(self) -> list[str]:
        """Return all tables in the configured dataset."""
        ds_ref = self._client.dataset(self.cfg.dataset)
        return sorted(t.table_id for t in self._client.list_tables(ds_ref))

    def table_info(self, table: str) -> TableMeta:
        """Return schema + row counts + partitioning for a single table."""
        full = self.cfg.fq_table(table)
        t = self._client.get_table(full)
        return TableMeta(
            table_id=t.table_id,
            full_table_id=full,
            num_rows=int(t.num_rows or 0),
            num_bytes=int(t.num_bytes or 0),
            created=t.created.isoformat() if t.created else None,
            modified=t.modified.isoformat() if t.modified else None,
            schema=[
                {
                    "name": f.name,
                    "type": f.field_type,
                    "mode": f.mode,
                    "description": f.description,
                }
                for f in t.schema
            ],
            partitioning=(
                {
                    "type": t.time_partitioning.type_,
                    "field": t.time_partitioning.field,
                    "expiration_ms": t.time_partitioning.expiration_ms,
                }
                if t.time_partitioning is not None
                else None
            ),
            clustering_fields=list(t.clustering_fields) if t.clustering_fields else None,
        )

    # ----- Querying -----

    def _job_config(self) -> bigquery.QueryJobConfig:
        return bigquery.QueryJobConfig(maximum_bytes_billed=self.bytes_budget)

    def query_rows(self, sql: str, *, max_results: int | None = None) -> list[dict[str, Any]]:
        """Run `sql` and return all rows as a list of dicts.

        Use only for small results; large scans should go through the Storage
        Read API (Phase 1).
        """
        log.debug("Running query (%d chars)", len(sql))
        job = self._client.query(sql, job_config=self._job_config())
        rows = job.result(max_results=max_results)
        out: list[dict[str, Any]] = []
        for row in rows:
            out.append({k: _to_jsonable(v) for k, v in dict(row).items()})
        return out

    def query_scalar(self, sql: str) -> Any:
        """Run a query that returns a single row, single column."""
        rows = self.query_rows(sql, max_results=1)
        if not rows:
            return None
        first = rows[0]
        if len(first) != 1:
            msg = f"query_scalar expected one column, got {list(first.keys())}"
            raise ValueError(msg)
        return next(iter(first.values()))

    def sample(self, table: str, n: int = 100) -> list[dict[str, Any]]:
        """Return `n` random-ish rows from `table` for inspection.

        Tries TABLESAMPLE first (cheapest, may not be supported on small or
        clustered tables), falls back to a probabilistic random-row sample.
        Auth / network errors are NOT swallowed.
        """
        fq = f"`{self.cfg.fq_table(table)}`"
        sql = f"SELECT * FROM {fq} TABLESAMPLE SYSTEM (1 PERCENT) LIMIT {int(n)}"
        try:
            return self.query_rows(sql)
        except gcp_exceptions.BadRequest as exc:
            log.warning(
                "TABLESAMPLE not supported on %s (%s); falling back to RAND() < 0.001",
                table,
                exc,
            )
            sql_fb = f"SELECT * FROM {fq} WHERE RAND() < 0.001 LIMIT {int(n)}"
            return self.query_rows(sql_fb)
        # Auth / 5xx / NotFound deliberately propagate.


def _to_jsonable(v: Any) -> Any:
    """Convert BQ row values to JSON-serializable Python types.

    Order matters: dates / Decimals / bytes need explicit handling before
    falling back to `str()` so we don't lose information.
    """
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if hasattr(v, "isoformat"):  # datetime, date, time
        return v.isoformat()
    if isinstance(v, Decimal):
        # JSON has no native decimal; string preserves precision and round-trips.
        return str(v)
    if isinstance(v, (bytes, bytearray)):
        return v.hex()
    if isinstance(v, (list, tuple)):
        return [_to_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: _to_jsonable(x) for k, x in v.items()}
    return str(v)
