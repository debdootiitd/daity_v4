
## C2_tgt10_stp05_h120_slip5

**Label**: tgt=1.00%  stop=0.50% (R:R=2.00)  entry_slip=5min  h_max=120min  label_cost=19bps
**Backtest barriers**: tgt=1.00%  stop=0.50% (R:R=2.00)  h_max=120min
**Train**: lr=1e-04  steps=12000  final_loss=nan  best_L@10=0.0000
**Backtest grid:**

| variant | cost | N | win% | AvgW bps | AvgL bps | Total % | Sharpe | DD % |
|---|---|---|---|---|---|---|---|---|
| top1_T0 | cost_inst | 621 | 6.0% | +1.29 | -0.63 | -3.23% | -5.22 | 3.22% |
| top1_T0 | cost_retail | 621 | 1.1% | +3.01 | -1.98 | -11.92% | -19.27 | 11.90% |
| top1_T0.20 | cost_inst | 5 | 20.0% | +0.53 | -0.24 | -0.00% | -0.29 | 0.01% |
| top1_T0.20 | cost_retail | 5 | 0.0% | +0.00 | -1.49 | -0.07% | -2.16 | 0.06% |
| top3 | cost_inst | 1863 | 21.2% | +5.09 | -2.15 | -11.46% | -0.79 | 13.35% |
| top3 | cost_retail | 1863 | 14.8% | +5.61 | -3.34 | -37.54% | -2.59 | 37.52% |
| top1_T0_ambigProp | cost_inst | 621 | 6.0% | +1.29 | -0.63 | -3.23% | -5.22 | 3.22% |
| top1_T0_ambigProp | cost_retail | 621 | 1.1% | +3.01 | -1.98 | -11.92% | -19.27 | 11.90% |
| top1_T0_ambigMid | cost_inst | 621 | 6.0% | +1.29 | -0.63 | -3.23% | -5.22 | 3.22% |
| top1_T0_ambigMid | cost_retail | 621 | 1.1% | +3.01 | -1.98 | -11.92% | -19.27 | 11.90% |
| top3_ambigProp | cost_inst | 1863 | 21.2% | +5.09 | -2.15 | -11.46% | -0.79 | 13.35% |
| top3_ambigProp | cost_retail | 1863 | 14.8% | +5.61 | -3.34 | -37.54% | -2.59 | 37.52% |

**Auto observations:**
- win rate range: 0.0% – 21.2%
- return range: -37.54% – -0.00%
- best cell: **top1_T0.20 / cost_inst** → -0.00% on 5 trades (20.0% win rate)
