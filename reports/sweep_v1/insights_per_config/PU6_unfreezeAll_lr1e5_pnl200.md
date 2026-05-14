
## PU6_unfreezeAll_lr1e5_pnl200

**Label**: tgt=0.50%  stop=0.30% (R:R=1.67)  entry_slip=5min  h_max=60min  label_cost=19bps
**Backtest barriers**: tgt=0.50%  stop=0.30% (R:R=1.67)  h_max=60min
**Train**: lr=1e-05  steps=20000  final_loss=nan  best_L@10=0.0000
**Backtest grid:**

| variant | cost | N | win% | AvgW bps | AvgL bps | Total % | Sharpe | DD % |
|---|---|---|---|---|---|---|---|---|
| top1_T0 | cost_inst | 621 | 38.8% | +5.96 | -5.34 | -5.91% | -0.59 | 8.19% |
| top1_T0 | cost_retail | 621 | 27.2% | +6.80 | -5.77 | -14.60% | -1.46 | 15.08% |
| top1_T0.20 | cost_inst | 621 | 38.8% | +5.96 | -5.34 | -5.91% | -0.59 | 8.19% |
| top1_T0.20 | cost_retail | 621 | 27.2% | +6.80 | -5.77 | -14.60% | -1.46 | 15.08% |
| top3 | cost_inst | 1863 | 39.0% | +4.21 | -4.04 | -15.22% | -1.16 | 16.14% |
| top3 | cost_retail | 1863 | 26.0% | +4.57 | -4.60 | -41.30% | -3.15 | 41.25% |
| top1_T0_ambigProp | cost_inst | 621 | 38.8% | +5.96 | -5.34 | -5.91% | -0.59 | 8.19% |
| top1_T0_ambigProp | cost_retail | 621 | 27.2% | +6.80 | -5.77 | -14.60% | -1.46 | 15.08% |
| top1_T0_ambigMid | cost_inst | 621 | 38.8% | +5.96 | -5.34 | -5.91% | -0.59 | 8.19% |
| top1_T0_ambigMid | cost_retail | 621 | 27.2% | +6.80 | -5.77 | -14.60% | -1.46 | 15.08% |
| top3_ambigProp | cost_inst | 1863 | 39.0% | +4.21 | -4.04 | -15.22% | -1.16 | 16.14% |
| top3_ambigProp | cost_retail | 1863 | 26.0% | +4.57 | -4.60 | -41.30% | -3.15 | 41.25% |

**Auto observations:**
- win rate range: 26.0% – 39.0%
- return range: -41.30% – -5.91%
- best cell: **top1_T0 / cost_inst** → -5.91% on 621 trades (38.8% win rate)
- ⚠️  top1_T0/cost_inst: realized R:R=1.12 + win<50% → structurally negative EV
