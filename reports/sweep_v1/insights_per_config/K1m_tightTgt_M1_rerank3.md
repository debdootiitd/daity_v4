
## K1m_tightTgt_M1_rerank3

**Label**: tgt=0.30%  stop=0.20% (R:R=1.50)  entry_slip=5min  h_max=60min  label_cost=19bps
**Backtest barriers**: tgt=0.30%  stop=0.20% (R:R=1.50)  h_max=60min
**Train**: lr=1e-04  steps=12000  final_loss=5.2172  best_L@10=0.1732
**Backtest grid:**

| variant | cost | N | win% | AvgW bps | AvgL bps | Total % | Sharpe | DD % |
|---|---|---|---|---|---|---|---|---|
| top1_T0 | cost_inst | 621 | 33.7% | +1.40 | -1.54 | -3.42% | -1.28 | 3.63% |
| top1_T0 | cost_retail | 621 | 15.8% | +0.95 | -2.49 | -12.11% | -4.53 | 12.08% |
| top1_T0.20 | cost_inst | 621 | 33.7% | +1.40 | -1.54 | -3.42% | -1.28 | 3.63% |
| top1_T0.20 | cost_retail | 621 | 15.8% | +0.95 | -2.49 | -12.11% | -4.53 | 12.08% |
| top3 | cost_inst | 1863 | 34.4% | +1.26 | -1.52 | -10.50% | -2.70 | 10.66% |
| top3 | cost_retail | 1863 | 13.7% | +0.94 | -2.43 | -36.58% | -9.42 | 36.56% |
| top1_T0_ambigProp | cost_inst | 621 | 33.7% | +1.40 | -1.54 | -3.42% | -1.28 | 3.63% |
| top1_T0_ambigProp | cost_retail | 621 | 15.8% | +0.95 | -2.49 | -12.11% | -4.53 | 12.08% |
| top1_T0_ambigMid | cost_inst | 621 | 33.7% | +1.40 | -1.54 | -3.42% | -1.28 | 3.63% |
| top1_T0_ambigMid | cost_retail | 621 | 15.8% | +0.95 | -2.49 | -12.11% | -4.53 | 12.08% |
| top3_ambigProp | cost_inst | 1863 | 34.4% | +1.26 | -1.52 | -10.50% | -2.70 | 10.66% |
| top3_ambigProp | cost_retail | 1863 | 13.7% | +0.94 | -2.43 | -36.58% | -9.42 | 36.56% |

**Auto observations:**
- win rate range: 13.7% – 34.4%
- return range: -36.58% – -3.42%
- best cell: **top1_T0 / cost_inst** → -3.42% on 621 trades (33.7% win rate)
- ⚠️  top1_T0/cost_inst: realized R:R=0.91 + win<50% → structurally negative EV
