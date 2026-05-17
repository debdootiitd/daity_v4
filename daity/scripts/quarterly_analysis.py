"""Quarter-wise split of all strategy results."""
import sys, os
sys.path.insert(0, "/root/daity"); os.chdir("/root/daity")
import polars as pl, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import date

RUNS = [
    ("Mode A static (d5)",            "reports/v12_strategy/gbm_k1_armA_N5.parquet",                  "today_pnl_bps",      5),
    ("Mode A static (neod)",          "reports/v12_strategy/gbm_k1_armA_neod_v2_N5.parquet",          "today_pnl_bps",      1),
    ("Mode A static + stop400 (neod)","reports/v12_strategy/gbm_k1_armA_neod_v2_N5_stop400.parquet",  "today_pnl_bps_stop", 1),
    ("Mode B adaptive (neod)",        "reports/v12_strategy/gbm_k1_armB_adagrad_v2_N5.parquet",       "today_pnl_bps",      1),
    ("Mode B adaptive + stop400 (neod)","reports/v12_strategy/gbm_k1_armB_adagrad_v2_N5_stop400.parquet","today_pnl_bps_stop",1),
]

def quarter_key(d):
    return f"{d.year}-Q{((d.month-1)//3)+1}"

def stats(bps, hold):
    n = len(bps)
    if n == 0: return None
    ann = float(np.sqrt(252.0 / hold))
    m = float(bps.mean()); sd = float(bps.std()) or 1e-9
    sleeve = float(np.prod(1 + bps/10000.0/hold) - 1) * 100
    nt = int((bps != 0).sum())
    hit = float((bps[bps != 0] > 0).sum() / max(nt, 1) * 100) if nt > 0 else 0.0
    return {"n": n, "trd": nt, "mean": m, "sharpe": m/sd*ann, "sleeve": sleeve, "hit": hit}

all_tables = {}
for label, path, col, hold in RUNS:
    df = pl.read_parquet(path).sort("test_date")
    qcol = [quarter_key(d) for d in df["test_date"].to_list()]
    df = df.with_columns(pl.Series("quarter", qcol))
    rows = []
    for q in sorted(set(qcol)):
        sub = df.filter(pl.col("quarter") == q)
        bps = sub[col].to_numpy()
        s = stats(bps, hold)
        if s is None: continue
        rows.append({"quarter": q, **s})
    bps_all = df[col].to_numpy()
    s_all = stats(bps_all, hold)
    rows.append({"quarter": "TOTAL", **s_all})
    all_tables[label] = rows

quarters = sorted(set(q for _, rows in all_tables.items() for q in [r["quarter"] for r in rows] if q != "TOTAL")) + ["TOTAL"]

for label, rows in all_tables.items():
    print(f"=== {label} ===")
    print(f"  {'q':<10}{'n':>5}{'trd':>5}{'mean_bps':>10}{'Sharpe':>9}{'sleeve%':>10}{'hit%':>8}")
    for r in rows:
        marker = " *" if r["quarter"] == "TOTAL" else ""
        print(f"  {r['quarter']:<10}{r['n']:>5}{r['trd']:>5}{r['mean']:>+10.2f}{r['sharpe']:>+9.2f}{r['sleeve']:>+10.2f}{r['hit']:>7.1f}%{marker}")
    print()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
labels_short = ["A.d5", "A.neod", "A.neod+stop", "B.neod", "B.neod+stop"]
colors = ["#7777aa", "#aaaaff", "#5555cc", "#22aa44", "#117733"]
quarters_only = [q for q in quarters if q != "TOTAL"]
x = np.arange(len(quarters_only))
width = 0.16
for i, (label, rows) in enumerate(all_tables.items()):
    sharpes = [next((r["sharpe"] for r in rows if r["quarter"] == q), 0.0) for q in quarters_only]
    sleeves = [next((r["sleeve"] for r in rows if r["quarter"] == q), 0.0) for q in quarters_only]
    ax1.bar(x + i*width - 2*width, sharpes, width, label=labels_short[i], color=colors[i], edgecolor="black", lw=0.3)
    ax2.bar(x + i*width - 2*width, sleeves, width, label=labels_short[i], color=colors[i], edgecolor="black", lw=0.3)

for ax, t, ylabel in [(ax1, "Sharpe per quarter (annualized, hold-adjusted)", "Sharpe"),
                      (ax2, "Sleeve return per quarter (%)", "Sleeve return (%)")]:
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(quarters_only)
    ax.set_title(t); ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=9, loc="best")
plt.tight_layout()
plt.savefig("reports/v12_strategy/quarterly_breakdown.png", dpi=120, bbox_inches="tight")
print("saved reports/v12_strategy/quarterly_breakdown.png")
