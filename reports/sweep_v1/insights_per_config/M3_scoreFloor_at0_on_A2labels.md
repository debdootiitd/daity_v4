
## M3_scoreFloor_at0_on_A2labels

**Label**: tgt=0.50%  stop=0.30% (R:R=1.67)  entry_slip=5min  h_max=60min  label_cost=19bps
**Backtest barriers**: tgt=0.50%  stop=0.30% (R:R=1.67)  h_max=60min
**Train**: lr=1e-04  steps=12000  final_loss=10.7093  best_L@10=0.1816
**Backtest grid:**

| variant | cost | N | win% | AvgW bps | AvgL bps | Total % | Sharpe | DD % |
|---|---|---|---|---|---|---|---|---|
| top1_T0 | cost_inst | 621 | 50.7% | +3.65 | -4.86 | -3.38% | -0.52 | 4.65% |
| top1_T0 | cost_retail | 621 | 43.5% | +2.73 | -5.54 | -12.08% | -1.84 | 12.55% |
| top1_T0.20 | cost_inst | 621 | 50.7% | +3.65 | -4.86 | -3.38% | -0.52 | 4.65% |
| top1_T0.20 | cost_retail | 621 | 43.5% | +2.73 | -5.54 | -12.08% | -1.84 | 12.55% |
| top3 | cost_inst | 1863 | 49.8% | +3.64 | -4.76 | -10.65% | -0.69 | 14.39% |
| top3 | cost_retail | 1863 | 42.5% | +2.76 | -5.47 | -36.73% | -2.39 | 37.70% |
| top1_T0_ambigProp | cost_inst | 621 | 50.7% | +3.65 | -4.86 | -3.38% | -0.52 | 4.65% |
| top1_T0_ambigProp | cost_retail | 621 | 43.5% | +2.73 | -5.54 | -12.08% | -1.84 | 12.55% |
| top1_T0_ambigMid | cost_inst | 621 | 50.7% | +3.65 | -4.86 | -3.38% | -0.52 | 4.65% |
| top1_T0_ambigMid | cost_retail | 621 | 43.5% | +2.73 | -5.54 | -12.08% | -1.84 | 12.55% |
| top3_ambigProp | cost_inst | 1863 | 49.8% | +3.64 | -4.76 | -10.65% | -0.69 | 14.39% |
| top3_ambigProp | cost_retail | 1863 | 42.5% | +2.76 | -5.47 | -36.73% | -2.39 | 37.70% |

**Auto observations:**
- win rate range: 42.5% – 50.7%
- return range: -36.73% – -3.38%
- best cell: **top1_T0 / cost_inst** → -3.38% on 621 trades (50.7% win rate)
- ⚠️  top1_T0/cost_retail: realized R:R=0.49 + win<50% → structurally negative EV
