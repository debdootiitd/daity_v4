
## M7_scoreRerank_K5_on_A2labels

**Label**: tgt=0.50%  stop=0.30% (R:R=1.67)  entry_slip=5min  h_max=60min  label_cost=19bps
**Backtest barriers**: tgt=0.50%  stop=0.30% (R:R=1.67)  h_max=60min
**Train**: lr=1e-04  steps=12000  final_loss=10.7093  best_L@10=0.1816
**Backtest grid:**

| variant | cost | N | win% | AvgW bps | AvgL bps | Total % | Sharpe | DD % |
|---|---|---|---|---|---|---|---|---|
| top1_T0 | cost_inst | 621 | 44.1% | +2.31 | -2.50 | -2.33% | -0.64 | 2.54% |
| top1_T0 | cost_retail | 621 | 24.2% | +2.31 | -3.08 | -11.03% | -3.04 | 10.95% |
| top1_T0.20 | cost_inst | 621 | 44.1% | +2.31 | -2.50 | -2.33% | -0.64 | 2.54% |
| top1_T0.20 | cost_retail | 621 | 24.2% | +2.31 | -3.08 | -11.03% | -3.04 | 10.95% |
| top3 | cost_inst | 1863 | 40.9% | +2.11 | -2.22 | -8.35% | -1.42 | 8.43% |
| top3 | cost_retail | 1863 | 21.5% | +2.08 | -2.93 | -34.43% | -5.85 | 34.32% |
| top1_T0_ambigProp | cost_inst | 621 | 44.1% | +2.31 | -2.50 | -2.33% | -0.64 | 2.54% |
| top1_T0_ambigProp | cost_retail | 621 | 24.2% | +2.31 | -3.08 | -11.03% | -3.04 | 10.95% |
| top1_T0_ambigMid | cost_inst | 621 | 44.1% | +2.31 | -2.50 | -2.33% | -0.64 | 2.54% |
| top1_T0_ambigMid | cost_retail | 621 | 24.2% | +2.31 | -3.08 | -11.03% | -3.04 | 10.95% |
| top3_ambigProp | cost_inst | 1863 | 40.9% | +2.11 | -2.22 | -8.35% | -1.42 | 8.43% |
| top3_ambigProp | cost_retail | 1863 | 21.5% | +2.08 | -2.93 | -34.43% | -5.85 | 34.32% |

**Auto observations:**
- win rate range: 21.5% – 44.1%
- return range: -34.43% – -2.33%
- best cell: **top1_T0 / cost_inst** → -2.33% on 621 trades (44.1% win rate)
- ⚠️  top1_T0/cost_inst: realized R:R=0.93 + win<50% → structurally negative EV
