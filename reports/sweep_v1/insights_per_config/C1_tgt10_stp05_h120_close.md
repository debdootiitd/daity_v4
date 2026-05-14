
## C1_tgt10_stp05_h120_close

**Label**: tgt=1.00%  stop=0.50% (R:R=2.00)  entry_slip=0min  h_max=120min  label_cost=19bps
**Backtest barriers**: tgt=1.00%  stop=0.50% (R:R=2.00)  h_max=120min
**Train**: lr=1e-04  steps=12000  final_loss=nan  best_L@10=0.0000
**Backtest grid:**

| variant | cost | N | win% | AvgW bps | AvgL bps | Total % | Sharpe | DD % |
|---|---|---|---|---|---|---|---|---|
| top1_T0 | cost_inst | 621 | 12.1% | +2.99 | -0.94 | -2.87% | -0.92 | 2.87% |
| top1_T0 | cost_retail | 621 | 4.8% | +5.32 | -2.23 | -11.56% | -3.72 | 11.54% |
| top1_T0.20 | cost_inst | 270 | 14.4% | +1.04 | -0.71 | -1.22% | -2.04 | 1.23% |
| top1_T0.20 | cost_retail | 270 | 1.9% | +3.72 | -1.96 | -5.00% | -4.52 | 4.98% |
| top3 | cost_inst | 1863 | 22.4% | +5.43 | -2.53 | -13.80% | -0.82 | 15.72% |
| top3 | cost_retail | 1863 | 16.6% | +5.73 | -3.71 | -39.88% | -2.38 | 39.86% |
| top1_T0_ambigProp | cost_inst | 621 | 12.1% | +2.99 | -0.94 | -2.87% | -0.92 | 2.87% |
| top1_T0_ambigProp | cost_retail | 621 | 4.8% | +5.32 | -2.23 | -11.56% | -3.72 | 11.54% |
| top1_T0_ambigMid | cost_inst | 621 | 12.1% | +2.99 | -0.94 | -2.87% | -0.92 | 2.87% |
| top1_T0_ambigMid | cost_retail | 621 | 4.8% | +5.32 | -2.23 | -11.56% | -3.72 | 11.54% |
| top3_ambigProp | cost_inst | 1863 | 22.4% | +5.43 | -2.53 | -13.80% | -0.82 | 15.72% |
| top3_ambigProp | cost_retail | 1863 | 16.6% | +5.73 | -3.71 | -39.88% | -2.38 | 39.86% |

**Auto observations:**
- win rate range: 1.9% – 22.4%
- return range: -39.88% – -1.22%
- best cell: **top1_T0.20 / cost_inst** → -1.22% on 270 trades (14.4% win rate)
