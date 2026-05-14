
## S5_trainWide_deployTight_stable

**Label**: tgt=0.75%  stop=0.35% (R:R=2.14)  entry_slip=0min  h_max=90min  label_cost=19bps
**Backtest barriers**: tgt=0.50%  stop=0.30% (R:R=1.67)  h_max=60min
**Train**: lr=3e-05  steps=18000  final_loss=5.2402  best_L@10=0.1580
**Backtest grid:**

| variant | cost | N | win% | AvgW bps | AvgL bps | Total % | Sharpe | DD % |
|---|---|---|---|---|---|---|---|---|
| top1_T0 | cost_inst | 621 | 33.7% | +3.69 | -3.18 | -5.40% | -1.67 | 5.70% |
| top1_T0 | cost_retail | 621 | 29.0% | +2.78 | -4.33 | -14.10% | -4.37 | 14.05% |
| top1_T0.20 | cost_inst | 0 | 0.0% | +0.00 | +0.00 | +0.00% | +0.00 | 0.00% |
| top1_T0.20 | cost_retail | 0 | 0.0% | +0.00 | +0.00 | +0.00% | +0.00 | 0.00% |
| top3 | cost_inst | 1863 | 35.5% | +3.70 | -3.19 | -13.86% | -1.89 | 14.23% |
| top3 | cost_retail | 1863 | 30.6% | +2.77 | -4.31 | -39.94% | -5.44 | 39.89% |
| top1_T0_ambigProp | cost_inst | 621 | 33.8% | +3.68 | -3.17 | -5.31% | -1.60 | 5.63% |
| top1_T0_ambigProp | cost_retail | 621 | 29.0% | +2.78 | -4.31 | -14.01% | -4.23 | 13.96% |
| top1_T0_ambigMid | cost_inst | 621 | 34.1% | +3.65 | -3.18 | -5.28% | -1.58 | 5.62% |
| top1_T0_ambigMid | cost_retail | 621 | 29.0% | +2.78 | -4.30 | -13.98% | -4.17 | 13.93% |
| top3_ambigProp | cost_inst | 1863 | 35.5% | +3.69 | -3.17 | -13.69% | -1.83 | 14.10% |
| top3_ambigProp | cost_retail | 1863 | 30.6% | +2.77 | -4.30 | -39.77% | -5.31 | 39.72% |

**Auto observations:**
- win rate range: 29.0% – 35.5%
- return range: -39.94% – -5.28%
- 2 variants returned 0 trades (score threshold too tight)
- best cell: **top1_T0.20 / cost_inst** → +0.00% on 0 trades (0.0% win rate)
- ⚠️  top1_T0/cost_inst: realized R:R=1.16 + win<50% → structurally negative EV
