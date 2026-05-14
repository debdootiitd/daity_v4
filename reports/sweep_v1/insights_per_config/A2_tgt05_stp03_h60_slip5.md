
## A2_tgt05_stp03_h60_slip5

**Label**: tgt=0.50%  stop=0.30% (R:R=1.67)  entry_slip=5min  h_max=60min  label_cost=19bps
**Backtest barriers**: tgt=0.50%  stop=0.30% (R:R=1.67)  h_max=60min
**Train**: lr=1e-04  steps=12000  final_loss=10.7093  best_L@10=0.1816
**Backtest grid:**

| variant | cost | N | win% | AvgW bps | AvgL bps | Total % | Sharpe | DD % |
|---|---|---|---|---|---|---|---|---|
| top1_T0 | cost_inst | 621 | 38.5% | +3.77 | -3.18 | -3.14% | -0.68 | 3.74% |
| top1_T0 | cost_retail | 621 | 33.3% | +2.84 | -4.28 | -11.84% | -2.56 | 11.82% |
| top1_T0.20 | cost_inst | 621 | 38.5% | +3.77 | -3.18 | -3.14% | -0.68 | 3.74% |
| top1_T0.20 | cost_retail | 621 | 33.3% | +2.84 | -4.28 | -11.84% | -2.56 | 11.82% |
| top3 | cost_inst | 1863 | 38.9% | +3.81 | -3.20 | -8.81% | -0.70 | 10.86% |
| top3 | cost_retail | 1863 | 34.4% | +2.82 | -4.33 | -34.89% | -2.77 | 34.85% |
| top1_T0_ambigProp | cost_inst | 621 | 38.8% | +3.76 | -3.18 | -3.03% | -0.68 | 3.63% |
| top1_T0_ambigProp | cost_retail | 621 | 33.7% | +2.82 | -4.28 | -11.73% | -2.62 | 11.70% |
| top1_T0_ambigMid | cost_inst | 621 | 38.8% | +3.74 | -3.18 | -3.06% | -0.68 | 3.66% |
| top1_T0_ambigMid | cost_retail | 621 | 33.3% | +2.84 | -4.26 | -11.76% | -2.60 | 11.74% |
| top3_ambigProp | cost_inst | 1863 | 39.1% | +3.80 | -3.20 | -8.59% | -0.69 | 10.64% |
| top3_ambigProp | cost_retail | 1863 | 34.5% | +2.81 | -4.33 | -34.67% | -2.79 | 34.62% |

**Auto observations:**
- win rate range: 33.3% – 39.1%
- return range: -34.89% – -3.03%
- best cell: **top1_T0_ambigProp / cost_inst** → -3.03% on 621 trades (38.8% win rate)
- ⚠️  top1_T0/cost_inst: realized R:R=1.18 + win<50% → structurally negative EV
