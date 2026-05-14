
## M6_scoreRerank_K4_on_A2labels

**Label**: tgt=0.50%  stop=0.30% (R:R=1.67)  entry_slip=5min  h_max=60min  label_cost=19bps
**Backtest barriers**: tgt=0.50%  stop=0.30% (R:R=1.67)  h_max=60min
**Train**: lr=1e-04  steps=12000  final_loss=10.7093  best_L@10=0.1816
**Backtest grid:**

| variant | cost | N | win% | AvgW bps | AvgL bps | Total % | Sharpe | DD % |
|---|---|---|---|---|---|---|---|---|
| top1_T0 | cost_inst | 621 | 43.2% | +2.27 | -2.36 | -2.25% | -0.61 | 2.50% |
| top1_T0 | cost_retail | 621 | 23.2% | +2.27 | -2.98 | -10.94% | -2.95 | 10.86% |
| top1_T0.20 | cost_inst | 621 | 43.2% | +2.27 | -2.36 | -2.25% | -0.61 | 2.50% |
| top1_T0.20 | cost_retail | 621 | 23.2% | +2.27 | -2.98 | -10.94% | -2.95 | 10.86% |
| top3 | cost_inst | 1863 | 40.3% | +1.98 | -2.11 | -8.61% | -1.48 | 8.63% |
| top3 | cost_retail | 1863 | 19.9% | +2.01 | -2.82 | -34.69% | -5.97 | 34.57% |
| top1_T0_ambigProp | cost_inst | 621 | 43.2% | +2.27 | -2.36 | -2.25% | -0.61 | 2.50% |
| top1_T0_ambigProp | cost_retail | 621 | 23.2% | +2.27 | -2.98 | -10.94% | -2.95 | 10.86% |
| top1_T0_ambigMid | cost_inst | 621 | 43.2% | +2.27 | -2.36 | -2.25% | -0.61 | 2.50% |
| top1_T0_ambigMid | cost_retail | 621 | 23.2% | +2.27 | -2.98 | -10.94% | -2.95 | 10.86% |
| top3_ambigProp | cost_inst | 1863 | 40.3% | +1.98 | -2.11 | -8.61% | -1.48 | 8.63% |
| top3_ambigProp | cost_retail | 1863 | 19.9% | +2.01 | -2.82 | -34.69% | -5.97 | 34.57% |

**Auto observations:**
- win rate range: 19.9% – 43.2%
- return range: -34.69% – -2.25%
- best cell: **top1_T0 / cost_inst** → -2.25% on 621 trades (43.2% win rate)
- ⚠️  top1_T0/cost_inst: realized R:R=0.96 + win<50% → structurally negative EV
