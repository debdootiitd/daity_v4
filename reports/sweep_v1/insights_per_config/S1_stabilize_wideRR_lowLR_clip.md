
## S1_stabilize_wideRR_lowLR_clip

**Label**: tgt=0.75%  stop=0.40% (R:R=1.88)  entry_slip=0min  h_max=60min  label_cost=19bps
**Backtest barriers**: tgt=0.75%  stop=0.40% (R:R=1.88)  h_max=60min
**Train**: lr=3e-05  steps=18000  final_loss=5.2222  best_L@10=0.1220
**Backtest grid:**

| variant | cost | N | win% | AvgW bps | AvgL bps | Total % | Sharpe | DD % |
|---|---|---|---|---|---|---|---|---|
| top1_T0 | cost_inst | 621 | 36.2% | +5.57 | -3.86 | -2.74% | -0.50 | 4.02% |
| top1_T0 | cost_retail | 621 | 31.6% | +4.92 | -4.96 | -11.44% | -2.07 | 12.03% |
| top1_T0.20 | cost_inst | 621 | 36.2% | +5.57 | -3.86 | -2.74% | -0.50 | 4.02% |
| top1_T0.20 | cost_retail | 621 | 31.6% | +4.92 | -4.96 | -11.44% | -2.07 | 12.03% |
| top3 | cost_inst | 1863 | 35.3% | +5.20 | -3.83 | -12.01% | -0.93 | 14.93% |
| top3 | cost_retail | 1863 | 30.3% | +4.55 | -4.91 | -38.09% | -2.96 | 38.73% |
| top1_T0_ambigProp | cost_inst | 621 | 36.4% | +5.56 | -3.85 | -2.64% | -0.46 | 3.92% |
| top1_T0_ambigProp | cost_retail | 621 | 31.7% | +4.90 | -4.95 | -11.33% | -1.99 | 11.92% |
| top1_T0_ambigMid | cost_inst | 621 | 36.6% | +5.53 | -3.86 | -2.63% | -0.46 | 3.91% |
| top1_T0_ambigMid | cost_retail | 621 | 31.6% | +4.92 | -4.93 | -11.32% | -1.98 | 11.91% |
| top3_ambigProp | cost_inst | 1863 | 35.4% | +5.19 | -3.82 | -11.84% | -0.92 | 14.76% |
| top3_ambigProp | cost_retail | 1863 | 30.3% | +4.54 | -4.90 | -37.92% | -2.93 | 38.55% |

**Auto observations:**
- win rate range: 30.3% – 36.6%
- return range: -38.09% – -2.63%
- best cell: **top1_T0_ambigMid / cost_inst** → -2.63% on 621 trades (36.6% win rate)
- ⚠️  top1_T0/cost_retail: realized R:R=0.99 + win<50% → structurally negative EV
