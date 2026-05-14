
## M4_scoreRerank_K1_aggressive

**Label**: tgt=0.50%  stop=0.30% (R:R=1.67)  entry_slip=5min  h_max=60min  label_cost=19bps
**Backtest barriers**: tgt=0.50%  stop=0.30% (R:R=1.67)  h_max=60min
**Train**: lr=1e-04  steps=12000  final_loss=10.7093  best_L@10=0.1816
**Backtest grid:**

| variant | cost | N | win% | AvgW bps | AvgL bps | Total % | Sharpe | DD % |
|---|---|---|---|---|---|---|---|---|
| top1_T0 | cost_inst | 621 | 39.1% | +1.73 | -1.82 | -2.70% | -1.20 | 2.72% |
| top1_T0 | cost_retail | 621 | 17.9% | +1.72 | -2.61 | -11.39% | -5.08 | 11.31% |
| top1_T0.20 | cost_inst | 621 | 39.1% | +1.73 | -1.82 | -2.70% | -1.20 | 2.72% |
| top1_T0.20 | cost_retail | 621 | 17.9% | +1.72 | -2.61 | -11.39% | -5.08 | 11.31% |
| top3 | cost_inst | 1863 | 38.2% | +1.51 | -1.70 | -8.80% | -2.03 | 8.70% |
| top3 | cost_retail | 1863 | 15.4% | +1.55 | -2.49 | -34.88% | -8.03 | 34.76% |
| top1_T0_ambigProp | cost_inst | 621 | 39.1% | +1.73 | -1.82 | -2.70% | -1.20 | 2.72% |
| top1_T0_ambigProp | cost_retail | 621 | 17.9% | +1.72 | -2.61 | -11.39% | -5.08 | 11.31% |
| top1_T0_ambigMid | cost_inst | 621 | 39.1% | +1.73 | -1.82 | -2.70% | -1.20 | 2.72% |
| top1_T0_ambigMid | cost_retail | 621 | 17.9% | +1.72 | -2.61 | -11.39% | -5.08 | 11.31% |
| top3_ambigProp | cost_inst | 1863 | 38.2% | +1.51 | -1.70 | -8.80% | -2.03 | 8.70% |
| top3_ambigProp | cost_retail | 1863 | 15.4% | +1.55 | -2.49 | -34.88% | -8.03 | 34.76% |

**Auto observations:**
- win rate range: 15.4% – 39.1%
- return range: -34.88% – -2.70%
- best cell: **top1_T0 / cost_inst** → -2.70% on 621 trades (39.1% win rate)
- ⚠️  top1_T0/cost_inst: realized R:R=0.95 + win<50% → structurally negative EV
