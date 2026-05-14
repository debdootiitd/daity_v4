
## PU4_unfreezeLast2_pnl500_T05_k5

**Label**: tgt=0.50%  stop=0.30% (R:R=1.67)  entry_slip=5min  h_max=60min  label_cost=19bps
**Backtest barriers**: tgt=0.50%  stop=0.30% (R:R=1.67)  h_max=60min
**Train**: lr=3e-05  steps=18000  final_loss=nan  best_L@10=0.0000
**Backtest grid:**

| variant | cost | N | win% | AvgW bps | AvgL bps | Total % | Sharpe | DD % |
|---|---|---|---|---|---|---|---|---|
| top1_T0 | cost_inst | 621 | 40.3% | +5.57 | -5.47 | -6.37% | -0.72 | 8.47% |
| top1_T0 | cost_retail | 621 | 29.1% | +6.01 | -5.90 | -15.07% | -1.70 | 15.12% |
| top1_T0.20 | cost_inst | 621 | 40.3% | +5.57 | -5.47 | -6.37% | -0.72 | 8.47% |
| top1_T0.20 | cost_retail | 621 | 29.1% | +6.01 | -5.90 | -15.07% | -1.70 | 15.12% |
| top3 | cost_inst | 1863 | 40.2% | +4.21 | -4.19 | -15.09% | -1.31 | 15.34% |
| top3 | cost_retail | 1863 | 27.2% | +4.48 | -4.71 | -41.18% | -3.57 | 41.12% |
| top1_T0_ambigProp | cost_inst | 621 | 40.3% | +5.57 | -5.47 | -6.37% | -0.72 | 8.47% |
| top1_T0_ambigProp | cost_retail | 621 | 29.1% | +6.01 | -5.90 | -15.07% | -1.70 | 15.12% |
| top1_T0_ambigMid | cost_inst | 621 | 40.3% | +5.57 | -5.47 | -6.37% | -0.72 | 8.47% |
| top1_T0_ambigMid | cost_retail | 621 | 29.1% | +6.01 | -5.90 | -15.07% | -1.70 | 15.12% |
| top3_ambigProp | cost_inst | 1863 | 40.2% | +4.21 | -4.19 | -15.09% | -1.31 | 15.34% |
| top3_ambigProp | cost_retail | 1863 | 27.2% | +4.48 | -4.71 | -41.18% | -3.57 | 41.12% |

**Auto observations:**
- win rate range: 27.2% – 40.3%
- return range: -41.18% – -6.37%
- best cell: **top1_T0 / cost_inst** → -6.37% on 621 trades (40.3% win rate)
- ⚠️  top1_T0/cost_inst: realized R:R=1.02 + win<50% → structurally negative EV
