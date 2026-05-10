"""Symbol master for the NSE universe.

Wraps `symbol_list` and `sector_metadata` (Phase 0 audit findings), caches the
joined view as Parquet, and centralizes universe identity. CLAUDE.md mandates
ISIN-based identity, but Phase 0 confirmed the BQ tables do not yet carry ISINs
(or `valid_from/valid_to`, `mcap_bucket`, `free_float_bucket`, `fno_eligible`,
`listed_on/delisted_on`). For now `symbol_id == symbol`, with an explicit flag
so callers can detect the temporary state. When ISINs land, the rebind happens
in this one file and the whole codebase picks it up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from daity.utils.logging import get_logger

if TYPE_CHECKING:
    from daity.data.bq import BQClient
    from daity.data.groww import GrowwClient

log = get_logger(__name__)

SYMBOL_LIST_TABLE = "symbol_list"
SECTOR_METADATA_TABLE = "sector_metadata"

CACHE_SUBDIR = "symbols"
SYMBOLS_PARQUET = "symbol_master.parquet"

# Columns we will eventually carry but do not have yet. We materialize them as
# all-null so callers can read a stable schema; `missing_columns` declares the
# gap so callers can refuse to act on them silently.
PLACEHOLDER_COLUMNS: tuple[str, ...] = (
    "isin",
    "valid_from",
    "valid_to",
    "mcap_bucket",
    "free_float_bucket",
    "fno_eligible",
    "listed_on",
    "delisted_on",
)


@dataclass(frozen=True, slots=True)
class SymbolMaster:
    """Joined view of `(symbol_list, sector_metadata)` plus placeholder columns.

    The frame is keyed by `symbol_id` (currently `== symbol`).

    `is_provisional` is True iff *any* identity column is still null —
    `missing_columns` lists the offenders. Even after ISIN backfill the
    master is provisional until the seven other identity columns
    (`valid_from / valid_to / mcap_bucket / free_float_bucket / fno_eligible /
    listed_on / delisted_on`) are populated. The flag is the coarse
    safety check; callers that want more granularity look at
    `missing_columns` directly. Use `has_column(name)` for "is X populated".
    """

    frame: pl.DataFrame
    missing_columns: tuple[str, ...] = field(default=PLACEHOLDER_COLUMNS)

    @property
    def is_provisional(self) -> bool:
        """True iff any required identity column is still missing.

        Computed from `missing_columns` rather than tracked separately so the
        invariant `is_provisional == bool(missing_columns)` cannot drift.
        """
        return bool(self.missing_columns)

    def has_column(self, name: str) -> bool:
        """True iff column `name` is populated in this master (i.e., not in
        `missing_columns`). Use this for column-specific gates rather than the
        coarse `is_provisional` flag.

        Example:
            if not sm.has_column("fno_eligible"):
                raise RuntimeError("Strategy needs F&O eligibility flag")
        """
        if name not in self.frame.columns:
            return False
        return name not in self.missing_columns

    @property
    def symbols(self) -> list[str]:
        return self.frame["symbol"].to_list()

    def __len__(self) -> int:
        return self.frame.height

    def lookup(self, symbol: str) -> dict | None:
        """Return the master row for `symbol` (or None)."""
        rows = self.frame.filter(pl.col("symbol") == symbol).to_dicts()
        return rows[0] if rows else None

    def in_sector(self, sector: str) -> list[str]:
        """All symbols in the given sector (case-insensitive)."""
        return (
            self.frame.filter(pl.col("sector").str.to_lowercase() == sector.lower())
            ["symbol"]
            .to_list()
        )

    def in_macro_sector(self, macro_sector: str) -> list[str]:
        """All symbols in the given macro sector (case-insensitive)."""
        return (
            self.frame.filter(pl.col("macro_sector").str.to_lowercase() == macro_sector.lower())
            ["symbol"]
            .to_list()
        )

    @classmethod
    def from_bq(
        cls,
        bq: BQClient,
        *,
        cache_root: Path | None = None,
        force_refresh: bool = False,
        groww_client: GrowwClient | None = None,
    ) -> SymbolMaster:
        """Pull from BQ, write Parquet cache, return the master.

        If `groww_client` is provided, the Groww instruments dump is also
        consulted to backfill the `isin` column (and `is_provisional` flips
        False if every symbol resolves). Without it, `isin` is null and
        `is_provisional` stays True.
        """
        if cache_root is not None and not force_refresh:
            cached = cls._try_from_cache(cache_root)
            if cached is not None:
                log.info("SymbolMaster: reusing cache at %s", cache_root)
                return cached

        log.info(
            "SymbolMaster: fetching %s + %s from BigQuery",
            SYMBOL_LIST_TABLE,
            SECTOR_METADATA_TABLE,
        )
        list_rows = bq.query_rows(
            f"SELECT symbol FROM `{bq.cfg.fq_table(SYMBOL_LIST_TABLE)}` ORDER BY symbol"
        )
        sector_rows = bq.query_rows(
            f"SELECT symbol, sector, industry, macro_sector "
            f"FROM `{bq.cfg.fq_table(SECTOR_METADATA_TABLE)}`"
        )

        list_df = pl.DataFrame(list_rows or [{"symbol": None}]).select(
            pl.col("symbol").cast(pl.Utf8)
        )
        sector_df = pl.DataFrame(
            sector_rows or [{"symbol": None, "sector": None, "industry": None, "macro_sector": None}]
        ).select(
            pl.col("symbol").cast(pl.Utf8),
            pl.col("sector").cast(pl.Utf8),
            pl.col("industry").cast(pl.Utf8),
            pl.col("macro_sector").cast(pl.Utf8),
        )

        isin_df: pl.DataFrame | None = None
        if groww_client is not None:
            symbols = list_df["symbol"].drop_nulls().to_list()
            log.info("SymbolMaster: backfilling ISINs from Groww for %d symbols", len(symbols))
            lookup = groww_client.bulk_lookup(symbols)
            isin_df = lookup.select(
                pl.col("trading_symbol").alias("symbol"),
                pl.col("isin").cast(pl.Utf8),
            )

        master = cls._build_frame(list_df, sector_df, isin_df=isin_df)
        if cache_root is not None:
            master._write_cache(cache_root)
        return master

    @classmethod
    def _build_frame(
        cls,
        list_df: pl.DataFrame,
        sector_df: pl.DataFrame,
        *,
        isin_df: pl.DataFrame | None = None,
    ) -> SymbolMaster:
        # Left join: every symbol in the universe gets sector info if present.
        joined = list_df.join(sector_df, on="symbol", how="left")
        n_with_sector = joined.filter(pl.col("sector").is_not_null()).height
        if n_with_sector < joined.height:
            log.warning(
                "SymbolMaster: %d / %d symbols missing sector metadata",
                joined.height - n_with_sector,
                joined.height,
            )

        # Decide ISIN handling: if a lookup table was provided, join it in;
        # otherwise leave isin null. The `missing_columns` set tracks which
        # identity columns are still null AT ALL — `is_provisional` is True
        # iff this set is non-empty, computed from `missing_columns` so it
        # cannot lie.
        missing_columns: tuple[str, ...] = PLACEHOLDER_COLUMNS
        if isin_df is not None:
            joined = joined.join(isin_df, on="symbol", how="left")
            n_with_isin = joined.filter(pl.col("isin").is_not_null()).height
            if n_with_isin == joined.height and joined.height > 0:
                missing_columns = tuple(c for c in PLACEHOLDER_COLUMNS if c != "isin")
                log.info(
                    "SymbolMaster: ISINs resolved for all %d symbols; "
                    "still missing: %s",
                    joined.height, list(missing_columns),
                )
            else:
                log.warning(
                    "SymbolMaster: ISINs resolved for %d / %d; "
                    "ISIN remains in missing_columns",
                    n_with_isin, joined.height,
                )
        else:
            joined = joined.with_columns(pl.lit(None).cast(pl.Utf8).alias("isin"))

        # Add provisional symbol_id (= symbol until ISINs become the identity)
        # and the remaining placeholder columns we still don't have.
        remaining_placeholders = [c for c in PLACEHOLDER_COLUMNS if c != "isin"]
        with_placeholders = joined.with_columns(
            [pl.col("symbol").alias("symbol_id")]
            + [pl.lit(None).alias(c) for c in remaining_placeholders]
        )
        # Order columns deterministically so cache + tests are stable.
        ordered = with_placeholders.select(
            ["symbol_id", "symbol", "sector", "industry", "macro_sector", "isin",
             *remaining_placeholders]
        )
        return cls(frame=ordered, missing_columns=missing_columns)

    @classmethod
    def from_cache(cls, cache_root: Path) -> SymbolMaster:
        cached = cls._try_from_cache(cache_root)
        if cached is None:
            msg = f"No symbol-master cache under {cache_root / CACHE_SUBDIR}"
            raise FileNotFoundError(msg)
        return cached

    @classmethod
    def _try_from_cache(cls, cache_root: Path) -> SymbolMaster | None:
        """Load the cached frame and *reconstruct* `missing_columns` from the
        data so the round-trip is honest about what's still null.

        For each placeholder column, we test the cached frame and drop it from
        `missing_columns` only if it's fully populated. `is_provisional` is
        derived from `missing_columns` so the two cannot drift.
        """
        d = Path(cache_root) / CACHE_SUBDIR
        p = d / SYMBOLS_PARQUET
        if not p.exists():
            return None
        frame = pl.read_parquet(p)
        # Reconstruct missing_columns: a column is "missing" if it isn't in
        # the frame OR has any null. This generalizes beyond the ISIN-only fix
        # so future backfills (mcap_bucket, fno_eligible, etc.) automatically
        # update the flag without code changes.
        missing: list[str] = []
        for col in PLACEHOLDER_COLUMNS:
            if col not in frame.columns or frame.height == 0 or frame[col].null_count() > 0:
                missing.append(col)
        return cls(frame=frame, missing_columns=tuple(missing))

    def _write_cache(self, cache_root: Path) -> None:
        d = Path(cache_root) / CACHE_SUBDIR
        d.mkdir(parents=True, exist_ok=True)
        self.frame.write_parquet(d / SYMBOLS_PARQUET)
        log.info("SymbolMaster: cached %d symbols at %s", self.frame.height, d)
