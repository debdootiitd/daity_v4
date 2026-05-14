
## M1_scoreRerank_K3_on_A2labels

**Label**: tgt=0.50%  stop=0.30% (R:R=1.67)  entry_slip=5min  h_max=60min  label_cost=19bps
**Backtest barriers**: tgt=0.50%  stop=0.30% (R:R=1.67)  h_max=60min
**Train**: lr=1e-04  steps=12000  final_loss=10.7093  best_L@10=0.1816
**Backtest grid:**

| variant | cost | N | win% | AvgW bps | AvgL bps | Total % | Sharpe | DD % |
|---|---|---|---|---|---|---|---|---|
| top1_T0 | cost_inst | 621 | 41.1% | +2.06 | -2.00 | -2.07% | -0.64 | 2.09% |
| top1_T0 | cost_retail | 621 | 20.6% | +2.08 | -2.72 | -10.77% | -3.32 | 10.68% |
| top1_T0.20 | cost_inst | 621 | 41.1% | +2.06 | -2.00 | -2.07% | -0.64 | 2.09% |
| top1_T0.20 | cost_retail | 621 | 20.6% | +2.08 | -2.72 | -10.77% | -3.32 | 10.68% |
| top3 | cost_inst | 1863 | 39.5% | +1.82 | -1.92 | -8.28% | -1.56 | 8.19% |
| top3 | cost_retail | 1863 | 18.5% | +1.83 | -2.68 | -34.37% | -6.49 | 34.25% |
| top1_T0_ambigProp | cost_inst | 621 | 41.1% | +2.06 | -2.00 | -2.07% | -0.64 | 2.09% |
| top1_T0_ambigProp | cost_retail | 621 | 20.6% | +2.08 | -2.72 | -10.77% | -3.32 | 10.68% |
| top1_T0_ambigMid | cost_inst | 621 | 41.1% | +2.06 | -2.00 | -2.07% | -0.64 | 2.09% |
| top1_T0_ambigMid | cost_retail | 621 | 20.6% | +2.08 | -2.72 | -10.77% | -3.32 | 10.68% |
| top3_ambigProp | cost_inst | 1863 | 39.5% | +1.82 | -1.92 | -8.28% | -1.56 | 8.19% |
| top3_ambigProp | cost_retail | 1863 | 18.5% | +1.83 | -2.68 | -34.37% | -6.49 | 34.25% |

**Auto observations:**
- win rate range: 18.5% – 41.1%
- return range: -34.37% – -2.07%
- best cell: **top1_T0 / cost_inst** → -2.07% on 621 trades (41.1% win rate)
- ⚠️  top1_T0/cost_inst: realized R:R=1.03 + win<50% → structurally negative EV
