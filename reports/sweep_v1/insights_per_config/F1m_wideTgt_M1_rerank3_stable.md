
## F1m_wideTgt_M1_rerank3_stable

**Label**: tgt=0.75%  stop=0.40% (R:R=1.88)  entry_slip=5min  h_max=180min  label_cost=19bps
**Backtest barriers**: tgt=0.75%  stop=0.40% (R:R=1.88)  h_max=180min
**Train**: lr=7e-05  steps=16000  final_loss=5.2449  best_L@10=0.1620
**Backtest grid:**

| variant | cost | N | win% | AvgW bps | AvgL bps | Total % | Sharpe | DD % |
|---|---|---|---|---|---|---|---|---|
| top1_T0 | cost_inst | 621 | 41.4% | +2.67 | -2.88 | -3.64% | -0.54 | 4.19% |
| top1_T0 | cost_retail | 621 | 23.0% | +2.88 | -3.44 | -12.34% | -1.84 | 12.45% |
| top1_T0.20 | cost_inst | 621 | 41.4% | +2.67 | -2.88 | -3.64% | -0.54 | 4.19% |
| top1_T0.20 | cost_retail | 621 | 23.0% | +2.88 | -3.44 | -12.34% | -1.84 | 12.45% |
| top3 | cost_inst | 1863 | 38.9% | +2.09 | -2.12 | -8.93% | -0.86 | 9.06% |
| top3 | cost_retail | 1863 | 18.0% | +2.43 | -2.83 | -35.02% | -3.35 | 35.08% |
| top1_T0_ambigProp | cost_inst | 621 | 41.4% | +2.67 | -2.88 | -3.64% | -0.54 | 4.19% |
| top1_T0_ambigProp | cost_retail | 621 | 23.0% | +2.88 | -3.44 | -12.34% | -1.84 | 12.45% |
| top1_T0_ambigMid | cost_inst | 621 | 41.4% | +2.67 | -2.88 | -3.64% | -0.54 | 4.19% |
| top1_T0_ambigMid | cost_retail | 621 | 23.0% | +2.88 | -3.44 | -12.34% | -1.84 | 12.45% |
| top3_ambigProp | cost_inst | 1863 | 38.9% | +2.09 | -2.12 | -8.93% | -0.86 | 9.06% |
| top3_ambigProp | cost_retail | 1863 | 18.0% | +2.43 | -2.83 | -35.02% | -3.35 | 35.08% |

**Auto observations:**
- win rate range: 18.0% – 41.4%
- return range: -35.02% – -3.64%
- best cell: **top1_T0 / cost_inst** → -3.64% on 621 trades (41.4% win rate)
- ⚠️  top1_T0/cost_inst: realized R:R=0.92 + win<50% → structurally negative EV
