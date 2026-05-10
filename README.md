# daity_v3

Foundation-model algo trading system for NSE equities. Pretrains a multi-resolution
patch transformer on OHLCV bars, post-trains supervised quantile / ranking heads with
conformal calibration, and fine-tunes the intraday head with order-book conditioning.

For the full design, see [`DESIGN.md`](DESIGN.md). The current implementation is in
**Phase 0** (bootstrap + data audit). See [`STATUS.md`](STATUS.md) for the live phase
ledger.

## Quickstart

```bash
# 1. Install
make dev                 # uv sync with dev + ml extras

# 2. Configure credentials (NEVER commit these files)
cp .env.example .env
# edit .env to point GOOGLE_APPLICATION_CREDENTIALS at your BQ service-account JSON

# Recommended: put the SA key OUTSIDE the repo tree to minimize blast radius
# (IDE syncs, agent sessions, Spotlight, Time Machine all touch the project dir).
mkdir -p ~/.config/gcloud && mv .sa-key.json ~/.config/gcloud/daity-sa.json
sed -i '' 's|GOOGLE_APPLICATION_CREDENTIALS=.*|GOOGLE_APPLICATION_CREDENTIALS=$HOME/.config/gcloud/daity-sa.json|' .env
chmod 600 ~/.config/gcloud/daity-sa.json
# If keeping the key in-repo for now: at minimum chmod 600 .sa-key.json
# (it's already gitignored; the chmod stops local exfiltration via tools that
# scan for world-readable secrets).

# Broker credentials (Phase 1+ ingest from Kite Connect):
# create .kite_cred.env at project root with KITE_API_KEY, KITE_API_SECRET,
# KITE_ACCESS_TOKEN. The access token rotates daily — `daity-kite-login`
# mints a fresh one. .kite_cred.env is gitignored (matches *cred* pattern).

# 3. Run the Phase 0 BigQuery data audit
make audit
# -> writes reports/phase0_audit.json + a console summary
```

## Layout

```
daity/        # library code (data / models / training / eval / strategy / serve)
configs/      # Hydra configs (data, model, train, strategy, backtest)
scripts/      # thin CLI entry points
tests/        # unit / leakage / smoke
notebooks/    # phase writeups, EDA, ablations
reports/      # generated audit reports + tearsheets
```

## Stack

PyTorch 2.x + Lightning, HuggingFace `peft` (LoRA), Polars + DuckDB + Parquet,
`google-cloud-bigquery-storage` (Arrow path), Hydra configs, W&B tracking, `uv` for env.

## Development

```bash
make test       # unit + leakage tests, no live BQ
make lint       # ruff + mypy
make format     # ruff format + autofix
```

Integration tests (`pytest -m integration`) hit live BigQuery and require
`GOOGLE_APPLICATION_CREDENTIALS` to be set.
