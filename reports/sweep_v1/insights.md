
## A1_tgt05_stp03_h60_close

**Label**: tgt=0.50%  stop=0.30% (R:R=1.67)  entry_slip=0min  h_max=60min  label_cost=19bps
**Backtest barriers**: tgt=0.50%  stop=0.30% (R:R=1.67)  h_max=60min
**Train**: lr=1e-04  steps=12000  final_loss=10.6701  best_L@10=0.1960
**Backtest grid:**

| variant | cost | N | win% | AvgW bps | AvgL bps | Total % | Sharpe | DD % |
|---|---|---|---|---|---|---|---|---|
| top1_T0 | cost_inst | 621 | 37.7% | +3.61 | -3.18 | -3.84% | -0.78 | 4.50% |
| top1_T0 | cost_retail | 621 | 32.0% | +2.74 | -4.26 | -12.53% | -2.54 | 12.64% |
| top1_T0.20 | cost_inst | 621 | 37.7% | +3.61 | -3.18 | -3.84% | -0.78 | 4.50% |
| top1_T0.20 | cost_retail | 621 | 32.0% | +2.74 | -4.26 | -12.53% | -2.54 | 12.64% |
| top3 | cost_inst | 1863 | 37.5% | +3.70 | -3.22 | -11.70% | -1.34 | 12.81% |
| top3 | cost_retail | 1863 | 32.4% | +2.79 | -4.33 | -37.78% | -4.32 | 37.74% |
| top1_T0_ambigProp | cost_inst | 621 | 37.7% | +3.61 | -3.17 | -3.83% | -0.78 | 4.49% |
| top1_T0_ambigProp | cost_retail | 621 | 32.0% | +2.74 | -4.26 | -12.52% | -2.54 | 12.63% |
| top1_T0_ambigMid | cost_inst | 621 | 37.8% | +3.60 | -3.17 | -3.80% | -0.77 | 4.46% |
| top1_T0_ambigMid | cost_retail | 621 | 32.0% | +2.74 | -4.25 | -12.49% | -2.52 | 12.60% |
| top3_ambigProp | cost_inst | 1863 | 37.5% | +3.70 | -3.22 | -11.61% | -1.33 | 12.71% |
| top3_ambigProp | cost_retail | 1863 | 32.4% | +2.78 | -4.33 | -37.69% | -4.30 | 37.64% |

**Auto observations:**
- win rate range: 32.0% – 37.8%
- return range: -37.78% – -3.80%
- best cell: **top1_T0_ambigMid / cost_inst** → -3.80% on 621 trades (37.8% win rate)
- ⚠️  top1_T0/cost_inst: realized R:R=1.14 + win<50% → structurally negative EV


## A2_tgt05_stp03_h60_slip5

**Label**: tgt=0.50%  stop=0.30% (R:R=1.67)  entry_slip=5min  h_max=60min  label_cost=19bps
**Backtest barriers**: tgt=0.50%  stop=0.30% (R:R=1.67)  h_max=60min
**Train**: lr=1e-04  steps=12000  final_loss=10.7093  best_L@10=0.1816
**Backtest grid:**

| variant | cost | N | win% | AvgW bps | AvgL bps | Total % | Sharpe | DD % |
|---|---|---|---|---|---|---|---|---|
| top1_T0 | cost_inst | 621 | 38.5% | +3.77 | -3.18 | -3.14% | -0.68 | 3.74% |
| top1_T0 | cost_retail | 621 | 33.3% | +2.84 | -4.28 | -11.84% | -2.56 | 11.82% |
| top1_T0.20 | cost_inst | 621 | 38.5% | +3.77 | -3.18 | -3.14% | -0.68 | 3.74% |
| top1_T0.20 | cost_retail | 621 | 33.3% | +2.84 | -4.28 | -11.84% | -2.56 | 11.82% |
| top3 | cost_inst | 1863 | 38.9% | +3.81 | -3.20 | -8.81% | -0.70 | 10.86% |
| top3 | cost_retail | 1863 | 34.4% | +2.82 | -4.33 | -34.89% | -2.77 | 34.85% |
| top1_T0_ambigProp | cost_inst | 621 | 38.8% | +3.76 | -3.18 | -3.03% | -0.68 | 3.63% |
| top1_T0_ambigProp | cost_retail | 621 | 33.7% | +2.82 | -4.28 | -11.73% | -2.62 | 11.70% |
| top1_T0_ambigMid | cost_inst | 621 | 38.8% | +3.74 | -3.18 | -3.06% | -0.68 | 3.66% |
| top1_T0_ambigMid | cost_retail | 621 | 33.3% | +2.84 | -4.26 | -11.76% | -2.60 | 11.74% |
| top3_ambigProp | cost_inst | 1863 | 39.1% | +3.80 | -3.20 | -8.59% | -0.69 | 10.64% |
| top3_ambigProp | cost_retail | 1863 | 34.5% | +2.81 | -4.33 | -34.67% | -2.79 | 34.62% |

**Auto observations:**
- win rate range: 33.3% – 39.1%
- return range: -34.89% – -3.03%
- best cell: **top1_T0_ambigProp / cost_inst** → -3.03% on 621 trades (38.8% win rate)
- ⚠️  top1_T0/cost_inst: realized R:R=1.18 + win<50% → structurally negative EV


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


## C1_tgt10_stp05_h120_close

**Label**: tgt=1.00%  stop=0.50% (R:R=2.00)  entry_slip=0min  h_max=120min  label_cost=19bps
**Backtest barriers**: tgt=1.00%  stop=0.50% (R:R=2.00)  h_max=120min
**Train**: lr=1e-04  steps=12000  final_loss=nan  best_L@10=0.0000
**Backtest grid:**

| variant | cost | N | win% | AvgW bps | AvgL bps | Total % | Sharpe | DD % |
|---|---|---|---|---|---|---|---|---|
| top1_T0 | cost_inst | 621 | 12.1% | +2.99 | -0.94 | -2.87% | -0.92 | 2.87% |
| top1_T0 | cost_retail | 621 | 4.8% | +5.32 | -2.23 | -11.56% | -3.72 | 11.54% |
| top1_T0.20 | cost_inst | 270 | 14.4% | +1.04 | -0.71 | -1.22% | -2.04 | 1.23% |
| top1_T0.20 | cost_retail | 270 | 1.9% | +3.72 | -1.96 | -5.00% | -4.52 | 4.98% |
| top3 | cost_inst | 1863 | 22.4% | +5.43 | -2.53 | -13.80% | -0.82 | 15.72% |
| top3 | cost_retail | 1863 | 16.6% | +5.73 | -3.71 | -39.88% | -2.38 | 39.86% |
| top1_T0_ambigProp | cost_inst | 621 | 12.1% | +2.99 | -0.94 | -2.87% | -0.92 | 2.87% |
| top1_T0_ambigProp | cost_retail | 621 | 4.8% | +5.32 | -2.23 | -11.56% | -3.72 | 11.54% |
| top1_T0_ambigMid | cost_inst | 621 | 12.1% | +2.99 | -0.94 | -2.87% | -0.92 | 2.87% |
| top1_T0_ambigMid | cost_retail | 621 | 4.8% | +5.32 | -2.23 | -11.56% | -3.72 | 11.54% |
| top3_ambigProp | cost_inst | 1863 | 22.4% | +5.43 | -2.53 | -13.80% | -0.82 | 15.72% |
| top3_ambigProp | cost_retail | 1863 | 16.6% | +5.73 | -3.71 | -39.88% | -2.38 | 39.86% |

**Auto observations:**
- win rate range: 1.9% – 22.4%
- return range: -39.88% – -1.22%
- best cell: **top1_T0.20 / cost_inst** → -1.22% on 270 trades (14.4% win rate)


## C2_tgt10_stp05_h120_slip5

**Label**: tgt=1.00%  stop=0.50% (R:R=2.00)  entry_slip=5min  h_max=120min  label_cost=19bps
**Backtest barriers**: tgt=1.00%  stop=0.50% (R:R=2.00)  h_max=120min
**Train**: lr=1e-04  steps=12000  final_loss=nan  best_L@10=0.0000
**Backtest grid:**

| variant | cost | N | win% | AvgW bps | AvgL bps | Total % | Sharpe | DD % |
|---|---|---|---|---|---|---|---|---|
| top1_T0 | cost_inst | 621 | 6.0% | +1.29 | -0.63 | -3.23% | -5.22 | 3.22% |
| top1_T0 | cost_retail | 621 | 1.1% | +3.01 | -1.98 | -11.92% | -19.27 | 11.90% |
| top1_T0.20 | cost_inst | 5 | 20.0% | +0.53 | -0.24 | -0.00% | -0.29 | 0.01% |
| top1_T0.20 | cost_retail | 5 | 0.0% | +0.00 | -1.49 | -0.07% | -2.16 | 0.06% |
| top3 | cost_inst | 1863 | 21.2% | +5.09 | -2.15 | -11.46% | -0.79 | 13.35% |
| top3 | cost_retail | 1863 | 14.8% | +5.61 | -3.34 | -37.54% | -2.59 | 37.52% |
| top1_T0_ambigProp | cost_inst | 621 | 6.0% | +1.29 | -0.63 | -3.23% | -5.22 | 3.22% |
| top1_T0_ambigProp | cost_retail | 621 | 1.1% | +3.01 | -1.98 | -11.92% | -19.27 | 11.90% |
| top1_T0_ambigMid | cost_inst | 621 | 6.0% | +1.29 | -0.63 | -3.23% | -5.22 | 3.22% |
| top1_T0_ambigMid | cost_retail | 621 | 1.1% | +3.01 | -1.98 | -11.92% | -19.27 | 11.90% |
| top3_ambigProp | cost_inst | 1863 | 21.2% | +5.09 | -2.15 | -11.46% | -0.79 | 13.35% |
| top3_ambigProp | cost_retail | 1863 | 14.8% | +5.61 | -3.34 | -37.54% | -2.59 | 37.52% |

**Auto observations:**
- win rate range: 0.0% – 21.2%
- return range: -37.54% – -0.00%
- best cell: **top1_T0.20 / cost_inst** → -0.00% on 5 trades (20.0% win rate)


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


## E1_wideRR_tgt08_stp03_h90

**Label**: tgt=0.80%  stop=0.30% (R:R=2.67)  entry_slip=5min  h_max=90min  label_cost=19bps
**Backtest barriers**: tgt=0.80%  stop=0.30% (R:R=2.67)  h_max=90min
**Train**: lr=1e-04  steps=12000  final_loss=nan  best_L@10=0.0000
**Backtest grid:**

| variant | cost | N | win% | AvgW bps | AvgL bps | Total % | Sharpe | DD % |
|---|---|---|---|---|---|---|---|---|
| top1_T0 | cost_inst | 621 | 2.7% | +0.31 | -0.53 | -3.12% | -8.46 | 3.12% |
| top1_T0 | cost_retail | 621 | 0.0% | +0.00 | -1.90 | -11.82% | -32.02 | 11.80% |
| top1_T0.20 | cost_inst | 233 | 6.4% | +0.30 | -0.53 | -1.10% | -2.20 | 1.10% |
| top1_T0.20 | cost_retail | 233 | 0.0% | +0.00 | -1.87 | -4.37% | -2.15 | 4.35% |
| top3 | cost_inst | 1863 | 20.3% | +3.55 | -1.58 | -10.04% | -1.62 | 10.59% |
| top3 | cost_retail | 1863 | 11.5% | +4.42 | -2.76 | -36.12% | -5.82 | 36.10% |
| top1_T0_ambigProp | cost_inst | 621 | 2.7% | +0.31 | -0.53 | -3.12% | -8.46 | 3.12% |
| top1_T0_ambigProp | cost_retail | 621 | 0.0% | +0.00 | -1.90 | -11.82% | -32.02 | 11.80% |
| top1_T0_ambigMid | cost_inst | 621 | 2.7% | +0.31 | -0.53 | -3.12% | -8.46 | 3.12% |
| top1_T0_ambigMid | cost_retail | 621 | 0.0% | +0.00 | -1.90 | -11.82% | -32.02 | 11.80% |
| top3_ambigProp | cost_inst | 1863 | 20.3% | +3.54 | -1.58 | -9.99% | -1.60 | 10.55% |
| top3_ambigProp | cost_retail | 1863 | 11.5% | +4.42 | -2.76 | -36.08% | -5.78 | 36.06% |

**Auto observations:**
- win rate range: 0.0% – 20.3%
- return range: -36.12% – -1.10%
- best cell: **top1_T0.20 / cost_inst** → -1.10% on 233 trades (6.4% win rate)
- ⚠️  top1_T0/cost_inst: realized R:R=0.59 + win<50% → structurally negative EV


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


## K1_tightTgt_retailLabelCost

**Label**: tgt=0.30%  stop=0.20% (R:R=1.50)  entry_slip=0min  h_max=30min  label_cost=19bps
**Backtest barriers**: tgt=0.30%  stop=0.20% (R:R=1.50)  h_max=30min
**Train**: lr=1e-04  steps=12000  final_loss=10.2976  best_L@10=0.1327
**Backtest grid:**

| variant | cost | N | win% | AvgW bps | AvgL bps | Total % | Sharpe | DD % |
|---|---|---|---|---|---|---|---|---|
| top1_T0 | cost_inst | 621 | 33.7% | +2.06 | -2.29 | -5.10% | -3.18 | 5.17% |
| top1_T0 | cost_retail | 621 | 26.7% | +1.05 | -3.41 | -13.80% | -8.59 | 13.76% |
| top1_T0.20 | cost_inst | 3 | 0.0% | +0.00 | -2.50 | -0.07% | +0.00 | 0.05% |
| top1_T0.20 | cost_retail | 3 | 0.0% | +0.00 | -3.90 | -0.12% | +0.00 | 0.08% |
| top3 | cost_inst | 1863 | 34.7% | +2.13 | -2.28 | -13.92% | -6.45 | 13.90% |
| top3 | cost_retail | 1863 | 28.6% | +1.06 | -3.43 | -40.01% | -18.52 | 39.97% |
| top1_T0_ambigProp | cost_inst | 621 | 33.7% | +2.06 | -2.28 | -5.07% | -3.11 | 5.14% |
| top1_T0_ambigProp | cost_retail | 621 | 26.7% | +1.05 | -3.41 | -13.76% | -8.43 | 13.73% |
| top1_T0_ambigMid | cost_inst | 621 | 33.7% | +2.06 | -2.27 | -5.05% | -3.08 | 5.12% |
| top1_T0_ambigMid | cost_retail | 621 | 26.7% | +1.05 | -3.40 | -13.75% | -8.38 | 13.71% |
| top3_ambigProp | cost_inst | 1863 | 34.7% | +2.13 | -2.27 | -13.79% | -6.29 | 13.77% |
| top3_ambigProp | cost_retail | 1863 | 28.6% | +1.06 | -3.42 | -39.87% | -18.19 | 39.84% |

**Auto observations:**
- win rate range: 0.0% – 34.7%
- return range: -40.01% – -0.07%
- best cell: **top1_T0.20 / cost_inst** → -0.07% on 3 trades (0.0% win rate)
- ⚠️  top1_T0/cost_inst: realized R:R=0.90 + win<50% → structurally negative EV


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


## M3_scoreFloor_at0_on_A2labels

**Label**: tgt=0.50%  stop=0.30% (R:R=1.67)  entry_slip=5min  h_max=60min  label_cost=19bps
**Backtest barriers**: tgt=0.50%  stop=0.30% (R:R=1.67)  h_max=60min
**Train**: lr=1e-04  steps=12000  final_loss=10.7093  best_L@10=0.1816
**Backtest grid:**

| variant | cost | N | win% | AvgW bps | AvgL bps | Total % | Sharpe | DD % |
|---|---|---|---|---|---|---|---|---|
| top1_T0 | cost_inst | 621 | 50.7% | +3.65 | -4.86 | -3.38% | -0.52 | 4.65% |
| top1_T0 | cost_retail | 621 | 43.5% | +2.73 | -5.54 | -12.08% | -1.84 | 12.55% |
| top1_T0.20 | cost_inst | 621 | 50.7% | +3.65 | -4.86 | -3.38% | -0.52 | 4.65% |
| top1_T0.20 | cost_retail | 621 | 43.5% | +2.73 | -5.54 | -12.08% | -1.84 | 12.55% |
| top3 | cost_inst | 1863 | 49.8% | +3.64 | -4.76 | -10.65% | -0.69 | 14.39% |
| top3 | cost_retail | 1863 | 42.5% | +2.76 | -5.47 | -36.73% | -2.39 | 37.70% |
| top1_T0_ambigProp | cost_inst | 621 | 50.7% | +3.65 | -4.86 | -3.38% | -0.52 | 4.65% |
| top1_T0_ambigProp | cost_retail | 621 | 43.5% | +2.73 | -5.54 | -12.08% | -1.84 | 12.55% |
| top1_T0_ambigMid | cost_inst | 621 | 50.7% | +3.65 | -4.86 | -3.38% | -0.52 | 4.65% |
| top1_T0_ambigMid | cost_retail | 621 | 43.5% | +2.73 | -5.54 | -12.08% | -1.84 | 12.55% |
| top3_ambigProp | cost_inst | 1863 | 49.8% | +3.64 | -4.76 | -10.65% | -0.69 | 14.39% |
| top3_ambigProp | cost_retail | 1863 | 42.5% | +2.76 | -5.47 | -36.73% | -2.39 | 37.70% |

**Auto observations:**
- win rate range: 42.5% – 50.7%
- return range: -36.73% – -3.38%
- best cell: **top1_T0 / cost_inst** → -3.38% on 621 trades (50.7% win rate)
- ⚠️  top1_T0/cost_retail: realized R:R=0.49 + win<50% → structurally negative EV


## M4_scoreRerank_K1_aggressive

**Label**: tgt=0.50%  stop=0.30% (R:R=1.67)  entry_slip=5min  h_max=60min  label_cost=19bps
**Backtest barriers**: tgt=0.50%  stop=0.30% (R:R=1.67)  h_max=60min
**Train**: lr=1e-04  steps=12000  final_loss=10.7093  best_L@10=0.1816
**Backtest grid:**

| variant | cost | N | win% | AvgW bps | AvgL bps | Total % | Sharpe | DD % |
|---|---|---|---|---|---|---|---|---|
| top1_T0 | cost_inst | 621 | 39.1% | +1.73 | -1.82 | -2.70% | -1.20 | 2.72% |
| top1_T0 | cost_retail | 621 | 17.9% | +1.72 | -2.61 | -11.39% | -5.08 | 11.31% |
| top1_T0.20 | cost_inst | 621 | 39.1% | +1.73 | -1.82 | -2.70% | -1.20 | 2.72% |
| top1_T0.20 | cost_retail | 621 | 17.9% | +1.72 | -2.61 | -11.39% | -5.08 | 11.31% |
| top3 | cost_inst | 1863 | 38.2% | +1.51 | -1.70 | -8.80% | -2.03 | 8.70% |
| top3 | cost_retail | 1863 | 15.4% | +1.55 | -2.49 | -34.88% | -8.03 | 34.76% |
| top1_T0_ambigProp | cost_inst | 621 | 39.1% | +1.73 | -1.82 | -2.70% | -1.20 | 2.72% |
| top1_T0_ambigProp | cost_retail | 621 | 17.9% | +1.72 | -2.61 | -11.39% | -5.08 | 11.31% |
| top1_T0_ambigMid | cost_inst | 621 | 39.1% | +1.73 | -1.82 | -2.70% | -1.20 | 2.72% |
| top1_T0_ambigMid | cost_retail | 621 | 17.9% | +1.72 | -2.61 | -11.39% | -5.08 | 11.31% |
| top3_ambigProp | cost_inst | 1863 | 38.2% | +1.51 | -1.70 | -8.80% | -2.03 | 8.70% |
| top3_ambigProp | cost_retail | 1863 | 15.4% | +1.55 | -2.49 | -34.88% | -8.03 | 34.76% |

**Auto observations:**
- win rate range: 15.4% – 39.1%
- return range: -34.88% – -2.70%
- best cell: **top1_T0 / cost_inst** → -2.70% on 621 trades (39.1% win rate)
- ⚠️  top1_T0/cost_inst: realized R:R=0.95 + win<50% → structurally negative EV


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


## M6_scoreRerank_K4_on_A2labels

**Label**: tgt=0.50%  stop=0.30% (R:R=1.67)  entry_slip=5min  h_max=60min  label_cost=19bps
**Backtest barriers**: tgt=0.50%  stop=0.30% (R:R=1.67)  h_max=60min
**Train**: lr=1e-04  steps=12000  final_loss=10.7093  best_L@10=0.1816
**Backtest grid:**

| variant | cost | N | win% | AvgW bps | AvgL bps | Total % | Sharpe | DD % |
|---|---|---|---|---|---|---|---|---|
| top1_T0 | cost_inst | 621 | 43.2% | +2.27 | -2.36 | -2.25% | -0.61 | 2.50% |
| top1_T0 | cost_retail | 621 | 23.2% | +2.27 | -2.98 | -10.94% | -2.95 | 10.86% |
| top1_T0.20 | cost_inst | 621 | 43.2% | +2.27 | -2.36 | -2.25% | -0.61 | 2.50% |
| top1_T0.20 | cost_retail | 621 | 23.2% | +2.27 | -2.98 | -10.94% | -2.95 | 10.86% |
| top3 | cost_inst | 1863 | 40.3% | +1.98 | -2.11 | -8.61% | -1.48 | 8.63% |
| top3 | cost_retail | 1863 | 19.9% | +2.01 | -2.82 | -34.69% | -5.97 | 34.57% |
| top1_T0_ambigProp | cost_inst | 621 | 43.2% | +2.27 | -2.36 | -2.25% | -0.61 | 2.50% |
| top1_T0_ambigProp | cost_retail | 621 | 23.2% | +2.27 | -2.98 | -10.94% | -2.95 | 10.86% |
| top1_T0_ambigMid | cost_inst | 621 | 43.2% | +2.27 | -2.36 | -2.25% | -0.61 | 2.50% |
| top1_T0_ambigMid | cost_retail | 621 | 23.2% | +2.27 | -2.98 | -10.94% | -2.95 | 10.86% |
| top3_ambigProp | cost_inst | 1863 | 40.3% | +1.98 | -2.11 | -8.61% | -1.48 | 8.63% |
| top3_ambigProp | cost_retail | 1863 | 19.9% | +2.01 | -2.82 | -34.69% | -5.97 | 34.57% |

**Auto observations:**
- win rate range: 19.9% – 43.2%
- return range: -34.69% – -2.25%
- best cell: **top1_T0 / cost_inst** → -2.25% on 621 trades (43.2% win rate)
- ⚠️  top1_T0/cost_inst: realized R:R=0.96 + win<50% → structurally negative EV


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


## PU1_unfreezeLast1_lr5e5_pnl200

**Label**: tgt=0.50%  stop=0.30% (R:R=1.67)  entry_slip=5min  h_max=60min  label_cost=19bps
**Backtest barriers**: tgt=0.50%  stop=0.30% (R:R=1.67)  h_max=60min
**Train**: lr=5e-05  steps=18000  final_loss=nan  best_L@10=0.0000
**Backtest grid:**

| variant | cost | N | win% | AvgW bps | AvgL bps | Total % | Sharpe | DD % |
|---|---|---|---|---|---|---|---|---|
| top1_T0 | cost_inst | 621 | 40.1% | +2.52 | -2.21 | -1.96% | -0.41 | 2.29% |
| top1_T0 | cost_retail | 621 | 20.6% | +2.86 | -2.90 | -10.66% | -2.22 | 10.57% |
| top1_T0.20 | cost_inst | 621 | 40.1% | +2.52 | -2.21 | -1.96% | -0.41 | 2.29% |
| top1_T0.20 | cost_retail | 621 | 20.6% | +2.86 | -2.90 | -10.66% | -2.22 | 10.57% |
| top3 | cost_inst | 1863 | 39.0% | +2.14 | -1.99 | -7.07% | -0.93 | 7.36% |
| top3 | cost_retail | 1863 | 17.6% | +2.58 | -2.71 | -33.15% | -4.34 | 33.10% |
| top1_T0_ambigProp | cost_inst | 621 | 40.1% | +2.52 | -2.21 | -1.96% | -0.41 | 2.29% |
| top1_T0_ambigProp | cost_retail | 621 | 20.6% | +2.86 | -2.90 | -10.66% | -2.22 | 10.57% |
| top1_T0_ambigMid | cost_inst | 621 | 40.1% | +2.52 | -2.21 | -1.96% | -0.41 | 2.29% |
| top1_T0_ambigMid | cost_retail | 621 | 20.6% | +2.86 | -2.90 | -10.66% | -2.22 | 10.57% |
| top3_ambigProp | cost_inst | 1863 | 39.0% | +2.14 | -1.99 | -7.07% | -0.93 | 7.36% |
| top3_ambigProp | cost_retail | 1863 | 17.6% | +2.58 | -2.71 | -33.15% | -4.34 | 33.10% |

**Auto observations:**
- win rate range: 17.6% – 40.1%
- return range: -33.15% – -1.96%
- best cell: **top1_T0 / cost_inst** → -1.96% on 621 trades (40.1% win rate)
- ⚠️  top1_T0/cost_inst: realized R:R=1.14 + win<50% → structurally negative EV


## PU2_unfreezeLast2_lr3e5_pnl200

**Label**: tgt=0.50%  stop=0.30% (R:R=1.67)  entry_slip=5min  h_max=60min  label_cost=19bps
**Backtest barriers**: tgt=0.50%  stop=0.30% (R:R=1.67)  h_max=60min
**Train**: lr=3e-05  steps=18000  final_loss=nan  best_L@10=0.0000
**Backtest grid:**

| variant | cost | N | win% | AvgW bps | AvgL bps | Total % | Sharpe | DD % |
|---|---|---|---|---|---|---|---|---|
| top1_T0 | cost_inst | 621 | 41.5% | +2.08 | -2.19 | -2.60% | -0.95 | 2.93% |
| top1_T0 | cost_retail | 621 | 20.3% | +2.13 | -2.82 | -11.29% | -4.13 | 11.22% |
| top1_T0.20 | cost_inst | 621 | 41.5% | +2.08 | -2.19 | -2.60% | -0.95 | 2.93% |
| top1_T0.20 | cost_retail | 621 | 20.3% | +2.13 | -2.82 | -11.29% | -4.13 | 11.22% |
| top3 | cost_inst | 1863 | 39.7% | +1.84 | -2.01 | -9.00% | -1.52 | 9.37% |
| top3 | cost_retail | 1863 | 18.0% | +1.88 | -2.71 | -35.08% | -5.92 | 35.05% |
| top1_T0_ambigProp | cost_inst | 621 | 41.5% | +2.08 | -2.19 | -2.60% | -0.95 | 2.93% |
| top1_T0_ambigProp | cost_retail | 621 | 20.3% | +2.13 | -2.82 | -11.29% | -4.13 | 11.22% |
| top1_T0_ambigMid | cost_inst | 621 | 41.5% | +2.08 | -2.19 | -2.60% | -0.95 | 2.93% |
| top1_T0_ambigMid | cost_retail | 621 | 20.3% | +2.13 | -2.82 | -11.29% | -4.13 | 11.22% |
| top3_ambigProp | cost_inst | 1863 | 39.7% | +1.84 | -2.01 | -9.00% | -1.52 | 9.37% |
| top3_ambigProp | cost_retail | 1863 | 18.0% | +1.88 | -2.71 | -35.08% | -5.92 | 35.05% |

**Auto observations:**
- win rate range: 18.0% – 41.5%
- return range: -35.08% – -2.60%
- best cell: **top1_T0 / cost_inst** → -2.60% on 621 trades (41.5% win rate)
- ⚠️  top1_T0/cost_inst: realized R:R=0.95 + win<50% → structurally negative EV


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


## PU4_unfreezeLast2_pnl500_T05_k5

**Label**: tgt=0.50%  stop=0.30% (R:R=1.67)  entry_slip=5min  h_max=60min  label_cost=19bps
**Backtest barriers**: tgt=0.50%  stop=0.30% (R:R=1.67)  h_max=60min
**Train**: lr=3e-05  steps=18000  final_loss=nan  best_L@10=0.0000
**Backtest grid:**

| variant | cost | N | win% | AvgW bps | AvgL bps | Total % | Sharpe | DD % |
|---|---|---|---|---|---|---|---|---|
| top1_T0 | cost_inst | 621 | 40.3% | +5.57 | -5.47 | -6.37% | -0.72 | 8.47% |
| top1_T0 | cost_retail | 621 | 29.1% | +6.01 | -5.90 | -15.07% | -1.70 | 15.12% |
| top1_T0.20 | cost_inst | 621 | 40.3% | +5.57 | -5.47 | -6.37% | -0.72 | 8.47% |
| top1_T0.20 | cost_retail | 621 | 29.1% | +6.01 | -5.90 | -15.07% | -1.70 | 15.12% |
| top3 | cost_inst | 1863 | 40.2% | +4.21 | -4.19 | -15.09% | -1.31 | 15.34% |
| top3 | cost_retail | 1863 | 27.2% | +4.48 | -4.71 | -41.18% | -3.57 | 41.12% |
| top1_T0_ambigProp | cost_inst | 621 | 40.3% | +5.57 | -5.47 | -6.37% | -0.72 | 8.47% |
| top1_T0_ambigProp | cost_retail | 621 | 29.1% | +6.01 | -5.90 | -15.07% | -1.70 | 15.12% |
| top1_T0_ambigMid | cost_inst | 621 | 40.3% | +5.57 | -5.47 | -6.37% | -0.72 | 8.47% |
| top1_T0_ambigMid | cost_retail | 621 | 29.1% | +6.01 | -5.90 | -15.07% | -1.70 | 15.12% |
| top3_ambigProp | cost_inst | 1863 | 40.2% | +4.21 | -4.19 | -15.09% | -1.31 | 15.34% |
| top3_ambigProp | cost_retail | 1863 | 27.2% | +4.48 | -4.71 | -41.18% | -3.57 | 41.12% |

**Auto observations:**
- win rate range: 27.2% – 40.3%
- return range: -41.18% – -6.37%
- best cell: **top1_T0 / cost_inst** → -6.37% on 621 trades (40.3% win rate)
- ⚠️  top1_T0/cost_inst: realized R:R=1.02 + win<50% → structurally negative EV


## PU5_unfreezeLast2_pnl100_T07_k20

**Label**: tgt=0.50%  stop=0.30% (R:R=1.67)  entry_slip=5min  h_max=60min  label_cost=19bps
**Backtest barriers**: tgt=0.50%  stop=0.30% (R:R=1.67)  h_max=60min
**Train**: lr=3e-05  steps=18000  final_loss=nan  best_L@10=0.0000
**Backtest grid:**

| variant | cost | N | win% | AvgW bps | AvgL bps | Total % | Sharpe | DD % |
|---|---|---|---|---|---|---|---|---|
| top1_T0 | cost_inst | 621 | 42.0% | +2.15 | -2.09 | -1.93% | -0.69 | 1.95% |
| top1_T0 | cost_retail | 621 | 20.8% | +2.25 | -2.75 | -10.62% | -3.77 | 10.54% |
| top1_T0.20 | cost_inst | 621 | 42.0% | +2.15 | -2.09 | -1.93% | -0.69 | 1.95% |
| top1_T0.20 | cost_retail | 621 | 20.8% | +2.25 | -2.75 | -10.62% | -3.77 | 10.54% |
| top3 | cost_inst | 1863 | 40.4% | +1.92 | -1.98 | -7.57% | -1.21 | 8.14% |
| top3 | cost_retail | 1863 | 18.5% | +2.04 | -2.68 | -33.65% | -5.37 | 33.62% |
| top1_T0_ambigProp | cost_inst | 621 | 42.0% | +2.15 | -2.09 | -1.93% | -0.69 | 1.95% |
| top1_T0_ambigProp | cost_retail | 621 | 20.8% | +2.25 | -2.75 | -10.62% | -3.77 | 10.54% |
| top1_T0_ambigMid | cost_inst | 621 | 42.0% | +2.15 | -2.09 | -1.93% | -0.69 | 1.95% |
| top1_T0_ambigMid | cost_retail | 621 | 20.8% | +2.25 | -2.75 | -10.62% | -3.77 | 10.54% |
| top3_ambigProp | cost_inst | 1863 | 40.4% | +1.92 | -1.98 | -7.57% | -1.21 | 8.14% |
| top3_ambigProp | cost_retail | 1863 | 18.5% | +2.04 | -2.68 | -33.65% | -5.37 | 33.62% |

**Auto observations:**
- win rate range: 18.5% – 42.0%
- return range: -33.65% – -1.93%
- best cell: **top1_T0 / cost_inst** → -1.93% on 621 trades (42.0% win rate)
- ⚠️  top1_T0/cost_inst: realized R:R=1.03 + win<50% → structurally negative EV


## PU6_unfreezeAll_lr1e5_pnl200

**Label**: tgt=0.50%  stop=0.30% (R:R=1.67)  entry_slip=5min  h_max=60min  label_cost=19bps
**Backtest barriers**: tgt=0.50%  stop=0.30% (R:R=1.67)  h_max=60min
**Train**: lr=1e-05  steps=20000  final_loss=nan  best_L@10=0.0000
**Backtest grid:**

| variant | cost | N | win% | AvgW bps | AvgL bps | Total % | Sharpe | DD % |
|---|---|---|---|---|---|---|---|---|
| top1_T0 | cost_inst | 621 | 38.8% | +5.96 | -5.34 | -5.91% | -0.59 | 8.19% |
| top1_T0 | cost_retail | 621 | 27.2% | +6.80 | -5.77 | -14.60% | -1.46 | 15.08% |
| top1_T0.20 | cost_inst | 621 | 38.8% | +5.96 | -5.34 | -5.91% | -0.59 | 8.19% |
| top1_T0.20 | cost_retail | 621 | 27.2% | +6.80 | -5.77 | -14.60% | -1.46 | 15.08% |
| top3 | cost_inst | 1863 | 39.0% | +4.21 | -4.04 | -15.22% | -1.16 | 16.14% |
| top3 | cost_retail | 1863 | 26.0% | +4.57 | -4.60 | -41.30% | -3.15 | 41.25% |
| top1_T0_ambigProp | cost_inst | 621 | 38.8% | +5.96 | -5.34 | -5.91% | -0.59 | 8.19% |
| top1_T0_ambigProp | cost_retail | 621 | 27.2% | +6.80 | -5.77 | -14.60% | -1.46 | 15.08% |
| top1_T0_ambigMid | cost_inst | 621 | 38.8% | +5.96 | -5.34 | -5.91% | -0.59 | 8.19% |
| top1_T0_ambigMid | cost_retail | 621 | 27.2% | +6.80 | -5.77 | -14.60% | -1.46 | 15.08% |
| top3_ambigProp | cost_inst | 1863 | 39.0% | +4.21 | -4.04 | -15.22% | -1.16 | 16.14% |
| top3_ambigProp | cost_retail | 1863 | 26.0% | +4.57 | -4.60 | -41.30% | -3.15 | 41.25% |

**Auto observations:**
- win rate range: 26.0% – 39.0%
- return range: -41.30% – -5.91%
- best cell: **top1_T0 / cost_inst** → -5.91% on 621 trades (38.8% win rate)
- ⚠️  top1_T0/cost_inst: realized R:R=1.12 + win<50% → structurally negative EV


## S1_stabilize_wideRR_lowLR_clip

**Label**: tgt=0.75%  stop=0.40% (R:R=1.88)  entry_slip=0min  h_max=60min  label_cost=19bps
**Backtest barriers**: tgt=0.75%  stop=0.40% (R:R=1.88)  h_max=60min
**Train**: lr=3e-05  steps=18000  final_loss=5.2222  best_L@10=0.1220
**Backtest grid:**

| variant | cost | N | win% | AvgW bps | AvgL bps | Total % | Sharpe | DD % |
|---|---|---|---|---|---|---|---|---|
| top1_T0 | cost_inst | 621 | 36.2% | +5.57 | -3.86 | -2.74% | -0.50 | 4.02% |
| top1_T0 | cost_retail | 621 | 31.6% | +4.92 | -4.96 | -11.44% | -2.07 | 12.03% |
| top1_T0.20 | cost_inst | 621 | 36.2% | +5.57 | -3.86 | -2.74% | -0.50 | 4.02% |
| top1_T0.20 | cost_retail | 621 | 31.6% | +4.92 | -4.96 | -11.44% | -2.07 | 12.03% |
| top3 | cost_inst | 1863 | 35.3% | +5.20 | -3.83 | -12.01% | -0.93 | 14.93% |
| top3 | cost_retail | 1863 | 30.3% | +4.55 | -4.91 | -38.09% | -2.96 | 38.73% |
| top1_T0_ambigProp | cost_inst | 621 | 36.4% | +5.56 | -3.85 | -2.64% | -0.46 | 3.92% |
| top1_T0_ambigProp | cost_retail | 621 | 31.7% | +4.90 | -4.95 | -11.33% | -1.99 | 11.92% |
| top1_T0_ambigMid | cost_inst | 621 | 36.6% | +5.53 | -3.86 | -2.63% | -0.46 | 3.91% |
| top1_T0_ambigMid | cost_retail | 621 | 31.6% | +4.92 | -4.93 | -11.32% | -1.98 | 11.91% |
| top3_ambigProp | cost_inst | 1863 | 35.4% | +5.19 | -3.82 | -11.84% | -0.92 | 14.76% |
| top3_ambigProp | cost_retail | 1863 | 30.3% | +4.54 | -4.90 | -37.92% | -2.93 | 38.55% |

**Auto observations:**
- win rate range: 30.3% – 36.6%
- return range: -38.09% – -2.63%
- best cell: **top1_T0_ambigMid / cost_inst** → -2.63% on 621 trades (36.6% win rate)
- ⚠️  top1_T0/cost_retail: realized R:R=0.99 + win<50% → structurally negative EV


## S5_trainWide_deployTight_stable

**Label**: tgt=0.75%  stop=0.35% (R:R=2.14)  entry_slip=0min  h_max=90min  label_cost=19bps
**Backtest barriers**: tgt=0.50%  stop=0.30% (R:R=1.67)  h_max=60min
**Train**: lr=3e-05  steps=18000  final_loss=5.2402  best_L@10=0.1580
**Backtest grid:**

| variant | cost | N | win% | AvgW bps | AvgL bps | Total % | Sharpe | DD % |
|---|---|---|---|---|---|---|---|---|
| top1_T0 | cost_inst | 621 | 33.7% | +3.69 | -3.18 | -5.40% | -1.67 | 5.70% |
| top1_T0 | cost_retail | 621 | 29.0% | +2.78 | -4.33 | -14.10% | -4.37 | 14.05% |
| top1_T0.20 | cost_inst | 0 | 0.0% | +0.00 | +0.00 | +0.00% | +0.00 | 0.00% |
| top1_T0.20 | cost_retail | 0 | 0.0% | +0.00 | +0.00 | +0.00% | +0.00 | 0.00% |
| top3 | cost_inst | 1863 | 35.5% | +3.70 | -3.19 | -13.86% | -1.89 | 14.23% |
| top3 | cost_retail | 1863 | 30.6% | +2.77 | -4.31 | -39.94% | -5.44 | 39.89% |
| top1_T0_ambigProp | cost_inst | 621 | 33.8% | +3.68 | -3.17 | -5.31% | -1.60 | 5.63% |
| top1_T0_ambigProp | cost_retail | 621 | 29.0% | +2.78 | -4.31 | -14.01% | -4.23 | 13.96% |
| top1_T0_ambigMid | cost_inst | 621 | 34.1% | +3.65 | -3.18 | -5.28% | -1.58 | 5.62% |
| top1_T0_ambigMid | cost_retail | 621 | 29.0% | +2.78 | -4.30 | -13.98% | -4.17 | 13.93% |
| top3_ambigProp | cost_inst | 1863 | 35.5% | +3.69 | -3.17 | -13.69% | -1.83 | 14.10% |
| top3_ambigProp | cost_retail | 1863 | 30.6% | +2.77 | -4.30 | -39.77% | -5.31 | 39.72% |

**Auto observations:**
- win rate range: 29.0% – 35.5%
- return range: -39.94% – -5.28%
- 2 variants returned 0 trades (score threshold too tight)
- best cell: **top1_T0.20 / cost_inst** → +0.00% on 0 trades (0.0% win rate)
- ⚠️  top1_T0/cost_inst: realized R:R=1.16 + win<50% → structurally negative EV
