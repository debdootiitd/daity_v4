
## PU1_unfreezeLast1_lr5e5_pnl200

**Label**: tgt=0.50%  stop=0.30% (R:R=1.67)  entry_slip=5min  h_max=60min  label_cost=19bps
**Backtest barriers**: tgt=0.50%  stop=0.30% (R:R=1.67)  h_max=60min
**Train**: lr=5e-05  steps=18000  final_loss=nan  best_L@10=0.0000
**Backtest grid:**

| variant | cost | N | win% | AvgW bps | AvgL bps | Total % | Sharpe | DD % |
|---|---|---|---|---|---|---|---|---|
| top1_T0 | cost_inst | 621 | 40.1% | +2.52 | -2.21 | -1.96% | -0.41 | 2.29% |
| top1_T0 | cost_retail | 621 | 20.6% | +2.86 | -2.90 | -10.66% | -2.22 | 10.57% |
| top1_T0.20 | cost_inst | 621 | 40.1% | +2.52 | -2.21 | -1.96% | -0.41 | 2.29% |
| top1_T0.20 | cost_retail | 621 | 20.6% | +2.86 | -2.90 | -10.66% | -2.22 | 10.57% |
| top3 | cost_inst | 1863 | 39.0% | +2.14 | -1.99 | -7.07% | -0.93 | 7.36% |
| top3 | cost_retail | 1863 | 17.6% | +2.58 | -2.71 | -33.15% | -4.34 | 33.10% |
| top1_T0_ambigProp | cost_inst | 621 | 40.1% | +2.52 | -2.21 | -1.96% | -0.41 | 2.29% |
| top1_T0_ambigProp | cost_retail | 621 | 20.6% | +2.86 | -2.90 | -10.66% | -2.22 | 10.57% |
| top1_T0_ambigMid | cost_inst | 621 | 40.1% | +2.52 | -2.21 | -1.96% | -0.41 | 2.29% |
| top1_T0_ambigMid | cost_retail | 621 | 20.6% | +2.86 | -2.90 | -10.66% | -2.22 | 10.57% |
| top3_ambigProp | cost_inst | 1863 | 39.0% | +2.14 | -1.99 | -7.07% | -0.93 | 7.36% |
| top3_ambigProp | cost_retail | 1863 | 17.6% | +2.58 | -2.71 | -33.15% | -4.34 | 33.10% |

**Auto observations:**
- win rate range: 17.6% – 40.1%
- return range: -33.15% – -1.96%
- best cell: **top1_T0 / cost_inst** → -1.96% on 621 trades (40.1% win rate)
- ⚠️  top1_T0/cost_inst: realized R:R=1.14 + win<50% → structurally negative EV
