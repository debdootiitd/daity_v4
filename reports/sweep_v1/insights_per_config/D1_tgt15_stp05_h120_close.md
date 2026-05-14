
## D1_tgt15_stp05_h120_close

**Label**: tgt=1.50%  stop=0.50% (R:R=3.00)  entry_slip=0min  h_max=120min  label_cost=19bps
**Backtest barriers**: tgt=1.50%  stop=0.50% (R:R=3.00)  h_max=120min
**Train**: lr=1e-04  steps=12000  final_loss=nan  best_L@10=0.0000
**Backtest grid:**

| variant | cost | N | win% | AvgW bps | AvgL bps | Total % | Sharpe | DD % |
|---|---|---|---|---|---|---|---|---|
| top1_T0 | cost_inst | 621 | 1.1% | +0.43 | -0.51 | -3.10% | -14.19 | 3.09% |
| top1_T0 | cost_retail | 621 | 0.0% | +0.00 | -1.90 | -11.79% | -53.99 | 11.77% |
| top1_T0.20 | cost_inst | 0 | 0.0% | +0.00 | +0.00 | +0.00% | +0.00 | 0.00% |
| top1_T0.20 | cost_retail | 0 | 0.0% | +0.00 | +0.00 | +0.00% | +0.00 | 0.00% |
| top3 | cost_inst | 1863 | 20.9% | +5.70 | -2.01 | -7.38% | -0.37 | 9.22% |
| top3 | cost_retail | 1863 | 13.9% | +6.84 | -3.19 | -33.47% | -1.67 | 33.45% |
| top1_T0_ambigProp | cost_inst | 621 | 1.1% | +0.43 | -0.51 | -3.10% | -14.19 | 3.09% |
| top1_T0_ambigProp | cost_retail | 621 | 0.0% | +0.00 | -1.90 | -11.79% | -53.99 | 11.77% |
| top1_T0_ambigMid | cost_inst | 621 | 1.1% | +0.43 | -0.51 | -3.10% | -14.19 | 3.09% |
| top1_T0_ambigMid | cost_retail | 621 | 0.0% | +0.00 | -1.90 | -11.79% | -53.99 | 11.77% |
| top3_ambigProp | cost_inst | 1863 | 20.9% | +5.70 | -2.01 | -7.38% | -0.37 | 9.22% |
| top3_ambigProp | cost_retail | 1863 | 13.9% | +6.84 | -3.19 | -33.47% | -1.67 | 33.45% |

**Auto observations:**
- win rate range: 0.0% – 20.9%
- return range: -33.47% – -3.10%
- 2 variants returned 0 trades (score threshold too tight)
- best cell: **top1_T0.20 / cost_inst** → +0.00% on 0 trades (0.0% win rate)
- ⚠️  top1_T0/cost_inst: realized R:R=0.85 + win<50% → structurally negative EV
