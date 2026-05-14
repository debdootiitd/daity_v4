
## PU3_lora_lr1e4_pnl200

**Label**: tgt=0.50%  stop=0.30% (R:R=1.67)  entry_slip=5min  h_max=60min  label_cost=19bps
**Backtest barriers**: tgt=0.50%  stop=0.30% (R:R=1.67)  h_max=60min
**Train**: lr=1e-04  steps=14000  final_loss=nan  best_L@10=0.0000
**Backtest grid:**

| variant | cost | N | win% | AvgW bps | AvgL bps | Total % | Sharpe | DD % |
|---|---|---|---|---|---|---|---|---|
| top1_T0 | cost_inst | 621 | 44.1% | +2.20 | -2.68 | -3.27% | -0.59 | 3.49% |
| top1_T0 | cost_retail | 621 | 21.6% | +2.39 | -3.12 | -11.96% | -2.17 | 11.87% |
| top1_T0.20 | cost_inst | 621 | 44.1% | +2.20 | -2.68 | -3.27% | -0.59 | 3.49% |
| top1_T0.20 | cost_retail | 621 | 21.6% | +2.39 | -3.12 | -11.96% | -2.17 | 11.87% |
| top3 | cost_inst | 1863 | 42.0% | +2.15 | -2.30 | -7.92% | -0.86 | 8.45% |
| top3 | cost_retail | 1863 | 20.0% | +2.44 | -2.89 | -34.00% | -3.67 | 33.91% |
| top1_T0_ambigProp | cost_inst | 621 | 44.1% | +2.20 | -2.68 | -3.27% | -0.59 | 3.49% |
| top1_T0_ambigProp | cost_retail | 621 | 21.6% | +2.39 | -3.12 | -11.96% | -2.17 | 11.87% |
| top1_T0_ambigMid | cost_inst | 621 | 44.1% | +2.20 | -2.68 | -3.27% | -0.59 | 3.49% |
| top1_T0_ambigMid | cost_retail | 621 | 21.6% | +2.39 | -3.12 | -11.96% | -2.17 | 11.87% |
| top3_ambigProp | cost_inst | 1863 | 42.0% | +2.15 | -2.30 | -7.92% | -0.86 | 8.45% |
| top3_ambigProp | cost_retail | 1863 | 20.0% | +2.44 | -2.89 | -34.00% | -3.67 | 33.91% |

**Auto observations:**
- win rate range: 20.0% – 44.1%
- return range: -34.00% – -3.27%
- best cell: **top1_T0 / cost_inst** → -3.27% on 621 trades (44.1% win rate)
- ⚠️  top1_T0/cost_inst: realized R:R=0.82 + win<50% → structurally negative EV
