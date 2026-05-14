
## PU5_unfreezeLast2_pnl100_T07_k20

**Label**: tgt=0.50%  stop=0.30% (R:R=1.67)  entry_slip=5min  h_max=60min  label_cost=19bps
**Backtest barriers**: tgt=0.50%  stop=0.30% (R:R=1.67)  h_max=60min
**Train**: lr=3e-05  steps=18000  final_loss=nan  best_L@10=0.0000
**Backtest grid:**

| variant | cost | N | win% | AvgW bps | AvgL bps | Total % | Sharpe | DD % |
|---|---|---|---|---|---|---|---|---|
| top1_T0 | cost_inst | 621 | 42.0% | +2.15 | -2.09 | -1.93% | -0.69 | 1.95% |
| top1_T0 | cost_retail | 621 | 20.8% | +2.25 | -2.75 | -10.62% | -3.77 | 10.54% |
| top1_T0.20 | cost_inst | 621 | 42.0% | +2.15 | -2.09 | -1.93% | -0.69 | 1.95% |
| top1_T0.20 | cost_retail | 621 | 20.8% | +2.25 | -2.75 | -10.62% | -3.77 | 10.54% |
| top3 | cost_inst | 1863 | 40.4% | +1.92 | -1.98 | -7.57% | -1.21 | 8.14% |
| top3 | cost_retail | 1863 | 18.5% | +2.04 | -2.68 | -33.65% | -5.37 | 33.62% |
| top1_T0_ambigProp | cost_inst | 621 | 42.0% | +2.15 | -2.09 | -1.93% | -0.69 | 1.95% |
| top1_T0_ambigProp | cost_retail | 621 | 20.8% | +2.25 | -2.75 | -10.62% | -3.77 | 10.54% |
| top1_T0_ambigMid | cost_inst | 621 | 42.0% | +2.15 | -2.09 | -1.93% | -0.69 | 1.95% |
| top1_T0_ambigMid | cost_retail | 621 | 20.8% | +2.25 | -2.75 | -10.62% | -3.77 | 10.54% |
| top3_ambigProp | cost_inst | 1863 | 40.4% | +1.92 | -1.98 | -7.57% | -1.21 | 8.14% |
| top3_ambigProp | cost_retail | 1863 | 18.5% | +2.04 | -2.68 | -33.65% | -5.37 | 33.62% |

**Auto observations:**
- win rate range: 18.5% – 42.0%
- return range: -33.65% – -1.93%
- best cell: **top1_T0 / cost_inst** → -1.93% on 621 trades (42.0% win rate)
- ⚠️  top1_T0/cost_inst: realized R:R=1.03 + win<50% → structurally negative EV
