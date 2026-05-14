
## M5_scoreRerank_K2_on_A2labels

**Label**: tgt=0.50%  stop=0.30% (R:R=1.67)  entry_slip=5min  h_max=60min  label_cost=19bps
**Backtest barriers**: tgt=0.50%  stop=0.30% (R:R=1.67)  h_max=60min
**Train**: lr=1e-04  steps=12000  final_loss=10.7093  best_L@10=0.1816
**Backtest grid:**

| variant | cost | N | win% | AvgW bps | AvgL bps | Total % | Sharpe | DD % |
|---|---|---|---|---|---|---|---|---|
| top1_T0 | cost_inst | 621 | 39.8% | +1.96 | -1.92 | -2.32% | -0.87 | 2.35% |
| top1_T0 | cost_retail | 621 | 19.5% | +1.95 | -2.68 | -11.02% | -4.11 | 10.93% |
| top1_T0.20 | cost_inst | 621 | 39.8% | +1.96 | -1.92 | -2.32% | -0.87 | 2.35% |
| top1_T0.20 | cost_retail | 621 | 19.5% | +1.95 | -2.68 | -11.02% | -4.11 | 10.93% |
| top3 | cost_inst | 1863 | 38.9% | +1.70 | -1.81 | -8.26% | -1.71 | 8.16% |
| top3 | cost_retail | 1863 | 16.9% | +1.76 | -2.58 | -34.34% | -7.13 | 34.22% |
| top1_T0_ambigProp | cost_inst | 621 | 39.8% | +1.96 | -1.92 | -2.32% | -0.87 | 2.35% |
| top1_T0_ambigProp | cost_retail | 621 | 19.5% | +1.95 | -2.68 | -11.02% | -4.11 | 10.93% |
| top1_T0_ambigMid | cost_inst | 621 | 39.8% | +1.96 | -1.92 | -2.32% | -0.87 | 2.35% |
| top1_T0_ambigMid | cost_retail | 621 | 19.5% | +1.95 | -2.68 | -11.02% | -4.11 | 10.93% |
| top3_ambigProp | cost_inst | 1863 | 38.9% | +1.70 | -1.81 | -8.26% | -1.71 | 8.16% |
| top3_ambigProp | cost_retail | 1863 | 16.9% | +1.76 | -2.58 | -34.34% | -7.13 | 34.22% |

**Auto observations:**
- win rate range: 16.9% – 39.8%
- return range: -34.34% – -2.32%
- best cell: **top1_T0 / cost_inst** → -2.32% on 621 trades (39.8% win rate)
- ⚠️  top1_T0/cost_inst: realized R:R=1.02 + win<50% → structurally negative EV
