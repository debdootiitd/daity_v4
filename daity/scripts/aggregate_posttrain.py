"""`daity-aggregate-posttrain` — collate per-run posttrain JSONs into the
two FoldHorizonResult-list files that `daity-walk-forward` reads.

Reads:
- `reports/phase3_posttrain_F-*_lora_seed0.json`     -> ssl_results
- `reports/phase3_posttrain_F-*_random_init_seed*.json` -> random_init_results
- `reports/phase3_xgboost_baseline.json`              -> xgb_results

Writes:
- `reports/phase3_ssl_results.json`
- `reports/phase3_random_init_results.json`
- `reports/phase3_xgb_results.json`

Each output is `{"results": [FoldHorizonResult dict, ...]}` — the format
`walk_forward_cli._load_fhr_json` expects.
"""

from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import click
from rich.console import Console

console = Console()

NAME_RE = re.compile(
    r"phase3_posttrain_(?P<fold>F-\d+)_(?P<mode>lora|random_init)_seed(?P<seed>\d+)\.json$"
)


def _to_fhr_dict(*, fold_id: str, horizon: str, mode: str, seed: int,
                 rank_ic_mean: float, n_cross_sections: int) -> dict:
    return {
        "fold_id": fold_id, "horizon": horizon, "mode": mode, "seed": seed,
        "rank_ic_mean": float(rank_ic_mean), "n_cross_sections": int(n_cross_sections),
    }


@click.command()
@click.option("--reports-dir", type=click.Path(path_type=Path),
              default=Path("reports"), show_default=True)
@click.option("--ssl-out", type=click.Path(path_type=Path),
              default=Path("reports/phase3_ssl_results.json"), show_default=True)
@click.option("--rand-out", type=click.Path(path_type=Path),
              default=Path("reports/phase3_random_init_results.json"), show_default=True)
@click.option("--xgb-out", type=click.Path(path_type=Path),
              default=Path("reports/phase3_xgb_results.json"), show_default=True)
@click.option("--xgb-input", type=click.Path(path_type=Path),
              default=Path("reports/phase3_xgboost_baseline.json"), show_default=True)
def main(reports_dir: Path, ssl_out: Path, rand_out: Path,
         xgb_out: Path, xgb_input: Path) -> None:
    """Walk reports/ and emit three FoldHorizonResult JSONs."""
    ssl_recs: list[dict] = []
    rand_recs: list[dict] = []
    for path in sorted(glob.glob(str(reports_dir / "phase3_posttrain_F-*.json"))):
        p = Path(path)
        if "smallval" in p.name: continue
        m = NAME_RE.search(p.name)
        if not m: continue
        fold = m.group("fold"); mode = m.group("mode"); seed = int(m.group("seed"))
        try: payload = json.loads(p.read_text())
        except Exception as e:
            console.print(f"[red]skip[/red] {p.name}: {e}"); continue
        ic_by_h = payload.get("per_horizon_rank_ic") or {}
        n_by_h  = payload.get("per_horizon_n_cross_sections") or {}
        for h, ic in ic_by_h.items():
            rec = _to_fhr_dict(
                fold_id=fold, horizon=h,
                mode=("ssl_heads" if mode == "lora" else "random_init"),
                seed=seed, rank_ic_mean=ic,
                n_cross_sections=int(n_by_h.get(h, 0)),
            )
            (ssl_recs if mode == "lora" else rand_recs).append(rec)

    xgb_recs: list[dict] = []
    if xgb_input.exists():
        xgb_doc = json.loads(xgb_input.read_text())
        for r in xgb_doc.get("results", []):
            xgb_recs.append(_to_fhr_dict(
                fold_id=r["fold_boundaries"]["fold_id"], horizon=r["horizon"],
                mode="xgboost", seed=0,
                rank_ic_mean=r["rank_ic"]["mean_ic"],
                n_cross_sections=r["rank_ic"]["n_cross_sections"],
            ))

    ssl_out.parent.mkdir(parents=True, exist_ok=True)
    ssl_out.write_text(json.dumps({"results": ssl_recs}, indent=2))
    rand_out.write_text(json.dumps({"results": rand_recs}, indent=2))
    xgb_out.write_text(json.dumps({"results": xgb_recs}, indent=2))

    console.print(f"[bold]Aggregated:[/bold]")
    console.print(f"  SSL+heads:    {len(ssl_recs):>3d} -> {ssl_out}")
    console.print(f"  random_init:  {len(rand_recs):>3d} -> {rand_out}")
    console.print(f"  XGBoost:      {len(xgb_recs):>3d} -> {xgb_out}")


if __name__ == "__main__":
    main()
