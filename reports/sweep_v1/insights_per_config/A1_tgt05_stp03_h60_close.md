
## A1_tgt05_stp03_h60_close

**Label**: tgt=0.50%  stop=0.30% (R:R=1.67)  entry_slip=0min  h_max=60min  label_cost=19bps
**Backtest barriers**: tgt=0.50%  stop=0.30% (R:R=1.67)  h_max=60min
**Train**: lr=1e-04  steps=12000  final_loss=10.6701  best_L@10=0.1960
**Backtest grid:**

| variant | cost | N | win% | AvgW bps | AvgL bps | Total % | Sharpe | DD % |
|---|---|---|---|---|---|---|---|---|
| top1_T0 | cost_inst | 621 | 37.7% | +3.61 | -3.18 | -3.84% | -0.78 | 4.50% |
| top1_T0 | cost_retail | 621 | 32.0% | +2.74 | -4.26 | -12.53% | -2.54 | 12.64% |
| top1_T0.20 | cost_inst | 621 | 37.7% | +3.61 | -3.18 | -3.84% | -0.78 | 4.50% |
| top1_T0.20 | cost_retail | 621 | 32.0% | +2.74 | -4.26 | -12.53% | -2.54 | 12.64% |
| top3 | cost_inst | 1863 | 37.5% | +3.70 | -3.22 | -11.70% | -1.34 | 12.81% |
| top3 | cost_retail | 1863 | 32.4% | +2.79 | -4.33 | -37.78% | -4.32 | 37.74% |
| top1_T0_ambigProp | cost_inst | 621 | 37.7% | +3.61 | -3.17 | -3.83% | -0.78 | 4.49% |
| top1_T0_ambigProp | cost_retail | 621 | 32.0% | +2.74 | -4.26 | -12.52% | -2.54 | 12.63% |
| top1_T0_ambigMid | cost_inst | 621 | 37.8% | +3.60 | -3.17 | -3.80% | -0.77 | 4.46% |
| top1_T0_ambigMid | cost_retail | 621 | 32.0% | +2.74 | -4.25 | -12.49% | -2.52 | 12.60% |
| top3_ambigProp | cost_inst | 1863 | 37.5% | +3.70 | -3.22 | -11.61% | -1.33 | 12.71% |
| top3_ambigProp | cost_retail | 1863 | 32.4% | +2.78 | -4.33 | -37.69% | -4.30 | 37.64% |

**Auto observations:**
- win rate range: 32.0% – 37.8%
- return range: -37.78% – -3.80%
- best cell: **top1_T0_ambigMid / cost_inst** → -3.80% on 621 trades (37.8% win rate)
- ⚠️  top1_T0/cost_inst: realized R:R=1.14 + win<50% → structurally negative EV
