
## K1_tightTgt_retailLabelCost

**Label**: tgt=0.30%  stop=0.20% (R:R=1.50)  entry_slip=0min  h_max=30min  label_cost=19bps
**Backtest barriers**: tgt=0.30%  stop=0.20% (R:R=1.50)  h_max=30min
**Train**: lr=1e-04  steps=12000  final_loss=10.2976  best_L@10=0.1327
**Backtest grid:**

| variant | cost | N | win% | AvgW bps | AvgL bps | Total % | Sharpe | DD % |
|---|---|---|---|---|---|---|---|---|
| top1_T0 | cost_inst | 621 | 33.7% | +2.06 | -2.29 | -5.10% | -3.18 | 5.17% |
| top1_T0 | cost_retail | 621 | 26.7% | +1.05 | -3.41 | -13.80% | -8.59 | 13.76% |
| top1_T0.20 | cost_inst | 3 | 0.0% | +0.00 | -2.50 | -0.07% | +0.00 | 0.05% |
| top1_T0.20 | cost_retail | 3 | 0.0% | +0.00 | -3.90 | -0.12% | +0.00 | 0.08% |
| top3 | cost_inst | 1863 | 34.7% | +2.13 | -2.28 | -13.92% | -6.45 | 13.90% |
| top3 | cost_retail | 1863 | 28.6% | +1.06 | -3.43 | -40.01% | -18.52 | 39.97% |
| top1_T0_ambigProp | cost_inst | 621 | 33.7% | +2.06 | -2.28 | -5.07% | -3.11 | 5.14% |
| top1_T0_ambigProp | cost_retail | 621 | 26.7% | +1.05 | -3.41 | -13.76% | -8.43 | 13.73% |
| top1_T0_ambigMid | cost_inst | 621 | 33.7% | +2.06 | -2.27 | -5.05% | -3.08 | 5.12% |
| top1_T0_ambigMid | cost_retail | 621 | 26.7% | +1.05 | -3.40 | -13.75% | -8.38 | 13.71% |
| top3_ambigProp | cost_inst | 1863 | 34.7% | +2.13 | -2.27 | -13.79% | -6.29 | 13.77% |
| top3_ambigProp | cost_retail | 1863 | 28.6% | +1.06 | -3.42 | -39.87% | -18.19 | 39.84% |

**Auto observations:**
- win rate range: 0.0% – 34.7%
- return range: -40.01% – -0.07%
- best cell: **top1_T0.20 / cost_inst** → -0.07% on 3 trades (0.0% win rate)
- ⚠️  top1_T0/cost_inst: realized R:R=0.90 + win<50% → structurally negative EV
