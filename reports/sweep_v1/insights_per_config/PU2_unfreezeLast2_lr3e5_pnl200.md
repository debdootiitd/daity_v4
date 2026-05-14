
## PU2_unfreezeLast2_lr3e5_pnl200

**Label**: tgt=0.50%  stop=0.30% (R:R=1.67)  entry_slip=5min  h_max=60min  label_cost=19bps
**Backtest barriers**: tgt=0.50%  stop=0.30% (R:R=1.67)  h_max=60min
**Train**: lr=3e-05  steps=18000  final_loss=nan  best_L@10=0.0000
**Backtest grid:**

| variant | cost | N | win% | AvgW bps | AvgL bps | Total % | Sharpe | DD % |
|---|---|---|---|---|---|---|---|---|
| top1_T0 | cost_inst | 621 | 41.5% | +2.08 | -2.19 | -2.60% | -0.95 | 2.93% |
| top1_T0 | cost_retail | 621 | 20.3% | +2.13 | -2.82 | -11.29% | -4.13 | 11.22% |
| top1_T0.20 | cost_inst | 621 | 41.5% | +2.08 | -2.19 | -2.60% | -0.95 | 2.93% |
| top1_T0.20 | cost_retail | 621 | 20.3% | +2.13 | -2.82 | -11.29% | -4.13 | 11.22% |
| top3 | cost_inst | 1863 | 39.7% | +1.84 | -2.01 | -9.00% | -1.52 | 9.37% |
| top3 | cost_retail | 1863 | 18.0% | +1.88 | -2.71 | -35.08% | -5.92 | 35.05% |
| top1_T0_ambigProp | cost_inst | 621 | 41.5% | +2.08 | -2.19 | -2.60% | -0.95 | 2.93% |
| top1_T0_ambigProp | cost_retail | 621 | 20.3% | +2.13 | -2.82 | -11.29% | -4.13 | 11.22% |
| top1_T0_ambigMid | cost_inst | 621 | 41.5% | +2.08 | -2.19 | -2.60% | -0.95 | 2.93% |
| top1_T0_ambigMid | cost_retail | 621 | 20.3% | +2.13 | -2.82 | -11.29% | -4.13 | 11.22% |
| top3_ambigProp | cost_inst | 1863 | 39.7% | +1.84 | -2.01 | -9.00% | -1.52 | 9.37% |
| top3_ambigProp | cost_retail | 1863 | 18.0% | +1.88 | -2.71 | -35.08% | -5.92 | 35.05% |

**Auto observations:**
- win rate range: 18.0% – 41.5%
- return range: -35.08% – -2.60%
- best cell: **top1_T0 / cost_inst** → -2.60% on 621 trades (41.5% win rate)
- ⚠️  top1_T0/cost_inst: realized R:R=0.95 + win<50% → structurally negative EV
