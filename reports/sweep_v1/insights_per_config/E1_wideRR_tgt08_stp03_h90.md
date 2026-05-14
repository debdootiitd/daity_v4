
## E1_wideRR_tgt08_stp03_h90

**Label**: tgt=0.80%  stop=0.30% (R:R=2.67)  entry_slip=5min  h_max=90min  label_cost=19bps
**Backtest barriers**: tgt=0.80%  stop=0.30% (R:R=2.67)  h_max=90min
**Train**: lr=1e-04  steps=12000  final_loss=nan  best_L@10=0.0000
**Backtest grid:**

| variant | cost | N | win% | AvgW bps | AvgL bps | Total % | Sharpe | DD % |
|---|---|---|---|---|---|---|---|---|
| top1_T0 | cost_inst | 621 | 2.7% | +0.31 | -0.53 | -3.12% | -8.46 | 3.12% |
| top1_T0 | cost_retail | 621 | 0.0% | +0.00 | -1.90 | -11.82% | -32.02 | 11.80% |
| top1_T0.20 | cost_inst | 233 | 6.4% | +0.30 | -0.53 | -1.10% | -2.20 | 1.10% |
| top1_T0.20 | cost_retail | 233 | 0.0% | +0.00 | -1.87 | -4.37% | -2.15 | 4.35% |
| top3 | cost_inst | 1863 | 20.3% | +3.55 | -1.58 | -10.04% | -1.62 | 10.59% |
| top3 | cost_retail | 1863 | 11.5% | +4.42 | -2.76 | -36.12% | -5.82 | 36.10% |
| top1_T0_ambigProp | cost_inst | 621 | 2.7% | +0.31 | -0.53 | -3.12% | -8.46 | 3.12% |
| top1_T0_ambigProp | cost_retail | 621 | 0.0% | +0.00 | -1.90 | -11.82% | -32.02 | 11.80% |
| top1_T0_ambigMid | cost_inst | 621 | 2.7% | +0.31 | -0.53 | -3.12% | -8.46 | 3.12% |
| top1_T0_ambigMid | cost_retail | 621 | 0.0% | +0.00 | -1.90 | -11.82% | -32.02 | 11.80% |
| top3_ambigProp | cost_inst | 1863 | 20.3% | +3.54 | -1.58 | -9.99% | -1.60 | 10.55% |
| top3_ambigProp | cost_retail | 1863 | 11.5% | +4.42 | -2.76 | -36.08% | -5.78 | 36.06% |

**Auto observations:**
- win rate range: 0.0% – 20.3%
- return range: -36.12% – -1.10%
- best cell: **top1_T0.20 / cost_inst** → -1.10% on 233 trades (6.4% win rate)
- ⚠️  top1_T0/cost_inst: realized R:R=0.59 + win<50% → structurally negative EV
