
## K5m_NIFTY50_M1_rerank3_top3

**Label**: tgt=0.40%  stop=0.25% (R:R=1.60)  entry_slip=5min  h_max=60min  label_cost=12bps
**Backtest barriers**: tgt=0.40%  stop=0.25% (R:R=1.60)  h_max=60min
**Train**: lr=1e-04  steps=12000  final_loss=5.2370  best_L@10=0.1860
**Backtest grid:**

| variant | cost | N | win% | AvgW bps | AvgL bps | Total % | Sharpe | DD % |
|---|---|---|---|---|---|---|---|---|
| top1_T0 | cost_inst | 427 | 38.4% | +0.66 | -0.74 | -0.86% | -1.26 | 0.95% |
| top1_T0 | cost_retail | 427 | 18.3% | +0.40 | -1.23 | -3.98% | -5.63 | 3.97% |
| top1_T0.20 | cost_inst | 30 | 26.7% | +1.09 | -1.10 | -0.16% | -0.86 | 0.16% |
| top1_T0.20 | cost_retail | 30 | 23.3% | +0.43 | -1.76 | -0.37% | -1.10 | 0.35% |
| top3 | cost_inst | 1281 | 35.9% | +0.66 | -0.68 | -2.52% | -2.04 | 2.65% |
| top3 | cost_retail | 1281 | 16.8% | +0.39 | -1.19 | -11.87% | -5.30 | 11.86% |
| top1_T0_ambigProp | cost_inst | 427 | 38.4% | +0.66 | -0.74 | -0.86% | -1.26 | 0.95% |
| top1_T0_ambigProp | cost_retail | 427 | 18.3% | +0.40 | -1.23 | -3.98% | -5.63 | 3.97% |
| top1_T0_ambigMid | cost_inst | 427 | 38.4% | +0.66 | -0.74 | -0.86% | -1.26 | 0.95% |
| top1_T0_ambigMid | cost_retail | 427 | 18.3% | +0.40 | -1.23 | -3.98% | -5.63 | 3.97% |
| top3_ambigProp | cost_inst | 1281 | 35.9% | +0.66 | -0.68 | -2.52% | -2.04 | 2.65% |
| top3_ambigProp | cost_retail | 1281 | 16.8% | +0.39 | -1.19 | -11.87% | -5.30 | 11.86% |

**Auto observations:**
- win rate range: 16.8% – 38.4%
- return range: -11.87% – -0.16%
- best cell: **top1_T0.20 / cost_inst** → -0.16% on 30 trades (26.7% win rate)
- ⚠️  top1_T0/cost_inst: realized R:R=0.89 + win<50% → structurally negative EV
