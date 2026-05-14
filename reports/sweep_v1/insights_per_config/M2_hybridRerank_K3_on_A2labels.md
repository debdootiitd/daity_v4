
## M2_hybridRerank_K3_on_A2labels

**Label**: tgt=0.50%  stop=0.30% (R:R=1.67)  entry_slip=5min  h_max=60min  label_cost=19bps
**Backtest barriers**: tgt=0.50%  stop=0.30% (R:R=1.67)  h_max=60min
**Train**: lr=1e-04  steps=12000  final_loss=10.7093  best_L@10=0.1816
**Backtest grid:**

| variant | cost | N | win% | AvgW bps | AvgL bps | Total % | Sharpe | DD % |
|---|---|---|---|---|---|---|---|---|
| top1_T0 | cost_inst | 621 | 38.6% | +2.03 | -1.90 | -2.39% | -0.87 | 2.38% |
| top1_T0 | cost_retail | 621 | 19.2% | +2.06 | -2.70 | -11.08% | -4.05 | 11.03% |
| top1_T0.20 | cost_inst | 621 | 38.6% | +2.03 | -1.90 | -2.39% | -0.87 | 2.38% |
| top1_T0.20 | cost_retail | 621 | 19.2% | +2.06 | -2.70 | -11.08% | -4.05 | 11.03% |
| top3 | cost_inst | 1863 | 38.1% | +1.80 | -1.78 | -7.76% | -1.57 | 7.73% |
| top3 | cost_retail | 1863 | 17.7% | +1.82 | -2.60 | -33.84% | -6.83 | 33.80% |
| top1_T0_ambigProp | cost_inst | 621 | 38.6% | +2.03 | -1.90 | -2.39% | -0.87 | 2.38% |
| top1_T0_ambigProp | cost_retail | 621 | 19.2% | +2.06 | -2.70 | -11.08% | -4.05 | 11.03% |
| top1_T0_ambigMid | cost_inst | 621 | 38.6% | +2.03 | -1.90 | -2.39% | -0.87 | 2.38% |
| top1_T0_ambigMid | cost_retail | 621 | 19.2% | +2.06 | -2.70 | -11.08% | -4.05 | 11.03% |
| top3_ambigProp | cost_inst | 1863 | 38.1% | +1.80 | -1.78 | -7.76% | -1.57 | 7.73% |
| top3_ambigProp | cost_retail | 1863 | 17.7% | +1.82 | -2.60 | -33.84% | -6.83 | 33.80% |

**Auto observations:**
- win rate range: 17.7% – 38.6%
- return range: -33.84% – -2.39%
- best cell: **top1_T0 / cost_inst** → -2.39% on 621 trades (38.6% win rate)
- ⚠️  top1_T0/cost_inst: realized R:R=1.07 + win<50% → structurally negative EV
