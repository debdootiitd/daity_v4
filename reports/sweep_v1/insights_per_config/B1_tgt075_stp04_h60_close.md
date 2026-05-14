
## B1_tgt075_stp04_h60_close

**Label**: tgt=0.75%  stop=0.40% (R:R=1.88)  entry_slip=0min  h_max=60min  label_cost=19bps
**Backtest barriers**: tgt=0.75%  stop=0.40% (R:R=1.88)  h_max=60min
**Train**: lr=1e-04  steps=12000  final_loss=nan  best_L@10=0.0000
**Backtest grid:**

| variant | cost | N | win% | AvgW bps | AvgL bps | Total % | Sharpe | DD % |
|---|---|---|---|---|---|---|---|---|
| top1_T0 | cost_inst | 621 | 35.9% | +5.19 | -4.03 | -4.46% | -0.86 | 4.68% |
| top1_T0 | cost_retail | 621 | 31.6% | +4.41 | -5.13 | -13.15% | -2.54 | 13.13% |
| top1_T0.20 | cost_inst | 0 | 0.0% | +0.00 | +0.00 | +0.00% | +0.00 | 0.00% |
| top1_T0.20 | cost_retail | 0 | 0.0% | +0.00 | +0.00 | +0.00% | +0.00 | 0.00% |
| top3 | cost_inst | 1863 | 36.3% | +5.30 | -3.99 | -11.43% | -0.96 | 12.79% |
| top3 | cost_retail | 1863 | 31.6% | +4.59 | -5.07 | -37.52% | -3.15 | 37.61% |
| top1_T0_ambigProp | cost_inst | 621 | 36.2% | +5.15 | -4.03 | -4.36% | -0.86 | 4.58% |
| top1_T0_ambigProp | cost_retail | 621 | 31.6% | +4.41 | -5.11 | -13.06% | -2.57 | 13.04% |
| top1_T0_ambigMid | cost_inst | 621 | 36.2% | +5.16 | -4.03 | -4.34% | -0.86 | 4.56% |
| top1_T0_ambigMid | cost_retail | 621 | 31.6% | +4.41 | -5.10 | -13.04% | -2.58 | 13.02% |
| top3_ambigProp | cost_inst | 1863 | 36.5% | +5.28 | -3.98 | -11.22% | -0.95 | 12.57% |
| top3_ambigProp | cost_retail | 1863 | 31.6% | +4.59 | -5.05 | -37.30% | -3.17 | 37.39% |

**Auto observations:**
- win rate range: 31.6% – 36.5%
- return range: -37.52% – -4.34%
- 2 variants returned 0 trades (score threshold too tight)
- best cell: **top1_T0.20 / cost_inst** → +0.00% on 0 trades (0.0% win rate)
- ⚠️  top1_T0/cost_retail: realized R:R=0.86 + win<50% → structurally negative EV
