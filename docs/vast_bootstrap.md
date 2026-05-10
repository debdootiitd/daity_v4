# Bootstrapping the Vast.ai box for Phase 2 pretraining

This is a one-shot guide for spinning up a fresh Vast.ai GPU instance and
launching the daity_v3 SSL pretraining run. The local Mac is the dev
environment; Vast is the training rig.

## SSH

```bash
ssh -p 25211 root@ssh4.vast.ai -L 8080:localhost:8080
```

The `-L 8080:localhost:8080` forwards a local port for W&B's local UI if
you want offline logging viewable from the laptop.

## One-time setup (run on Vast)

```bash
# 1. System deps. Most Vast images already have CUDA + Python 3.11 — verify.
nvidia-smi                                   # confirms GPU + driver
python3.11 --version                         # need 3.11.x

# 2. uv (fast python env manager — what the project uses).
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# 3. Clone (or rsync from local — see below).
mkdir -p ~/work && cd ~/work
# If pushing from local:
#   rsync -avz --exclude='.venv' --exclude='data/' --exclude='.git' \
#     -e 'ssh -p 25211' /Users/debdoot/Documents/Claude/Projects/daity/ \
#     root@ssh4.vast.ai:~/work/daity/
# Or if pulled from a git remote:
git clone <repo> daity
cd daity

# 4. Install Python deps with the ml extra.
uv sync --extra dev --extra ml               # pulls torch+CUDA, lightning, wandb
```

## Credentials (run on Vast)

Three secrets land at the project root, all gitignored:

```bash
# 1. GCP service account JSON for BigQuery + GCS.
mkdir -p ~/.config/gcloud
# scp from your laptop:
#   scp -P 25211 ~/.config/gcloud/daity-sa.json root@ssh4.vast.ai:~/.config/gcloud/
chmod 600 ~/.config/gcloud/daity-sa.json

# 2. .env pointing at the SA key.
cat > ~/work/daity/.env <<'EOF'
GOOGLE_APPLICATION_CREDENTIALS=$HOME/.config/gcloud/daity-sa.json
DAITY_BQ_PROJECT=atomic-vault-476315-m3
DAITY_BQ_DATASET=daity_v3
EOF
chmod 600 ~/work/daity/.env

# 3. Kite credentials (Phase 1+ ingest only — skip if you rsync data/parquet/).
# scp .kite_cred.env from laptop similarly. Kite access tokens rotate daily;
# if you'll re-mint here, run `daity-kite-login` after the daily expiry.
```

## Data — two paths

### Path A: rsync the local Parquet cache (fastest first time)

The local `data/parquet/` is ~735 MB. Pushing it skips re-running ingest
on Vast:

```bash
# From laptop:
rsync -avz --progress -e 'ssh -p 25211' \
  /Users/debdoot/Documents/Claude/Projects/daity/data/parquet/ \
  root@ssh4.vast.ai:~/work/daity/data/parquet/
# Also push the symbol-master + calendar caches:
rsync -avz -e 'ssh -p 25211' \
  /Users/debdoot/Documents/Claude/Projects/daity/data/cache/ \
  root@ssh4.vast.ai:~/work/daity/data/cache/
```

### Path B: re-ingest from BigQuery on Vast (clean-room reproducible)

```bash
cd ~/work/daity
uv run daity-check-staleness            # confirm raw_ohlcv is fresh
uv run daity-ingest-parquet             # ~2 min via GCS export → split locally
```

## Smoke test (run on Vast)

Before kicking off the multi-day run, sanity-check:

```bash
cd ~/work/daity
# 100-step CPU smoke, completes in seconds — same path the Mac runs locally.
uv run daity-pretrain --smoke

# Full-config but tiny step count, on the GPU:
uv run daity-pretrain --max-steps 200 --device cuda --precision bf16-mixed
```

Tail the CSV log to confirm losses are decreasing:

```bash
tail -f runs/pretrain/lightning_logs/version_*/metrics.csv
```

## Full pretrain run (run on Vast, in tmux/screen)

```bash
# Inside tmux so the run survives SSH disconnects.
tmux new -s pretrain
cd ~/work/daity

# Optional: enable live W&B logging.
# wandb login                            # paste your API key once
# Then in configs/train/pretrain.yaml: wandb_offline: false

uv run daity-pretrain                    # uses configs/train/pretrain.yaml
```

DESIGN §3.2 expects ~2-3 days at the default config (50K steps, batch 64,
A100-40GB, bf16 mixed precision). Checkpoints land under `runs/pretrain/`
every 2 hours.

## Pulling artifacts back to the laptop

```bash
# After the run, sync the best 3 checkpoints + W&B logs back:
rsync -avz -e 'ssh -p 25211' \
  root@ssh4.vast.ai:~/work/daity/runs/pretrain/ \
  /Users/debdoot/Documents/Claude/Projects/daity/runs/pretrain/
```

Then run the Phase 2 writeup notebook (`notebooks/phase_2_writeup.py`)
locally to evaluate the checkpoint and produce the acceptance plots.

## Common gotchas

- **CUDA OOM** on 24 GB / 16 GB cards: reduce `batch_size` to 32 or 16,
  enable `gradient_checkpointing` (added in Phase 2.5 if needed). The
  default config targets A100-40GB.
- **Kite token expired mid-run**: only matters during ingest; pretraining
  reads only from local Parquet. Re-mint with `daity-kite-login` if a
  fresh ingest is needed.
- **`bf16-mixed` not supported** on older GPUs (T4, P100): fall back to
  `--precision 16-mixed` or `32-true`.
- **W&B offline mode**: with `wandb_offline: true` in the config, runs
  log to `runs/pretrain/wandb/offline-*` and can be uploaded later via
  `wandb sync`.
