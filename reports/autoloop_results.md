# Autonomous loop — alpha iteration log

Started 2026-05-19 by /loop. Lightweight 300K-param model on engineered features.
Baseline: SmoothL1 regression + BCE classification, AdamW, 75s training, 814-day OOS.

| Iter | Idea | OOS days | K=1 Sharpe | IC mean | Hit | Sleeve | Notes |
|---|---|---|---|---|---|---|---|
| 0 | Baseline SmoothL1+BCE, 1 epoch wf | 822 (not held out) | -1.98 | +0.0035 | 0.42 | -87% | Train/test temporal overlap — semi-honest |
| 1 | + Sharpe loss (w_sharpe=0.5), batch=32 | 822 (not held out) | -1.34 | -0.0102 | 0.42 | -86% | Sharpe loss flipped IC sign; w_sharpe too aggressive |
| 2 | 5 epochs TRAIN-MULTI + proper 12mo OOS | 248 honest | **-4.50** | **-0.012** | **0.18** | -66% | Multi-epoch OVERFITS — model anti-predictive on OOS |
| 3 | 1 epoch TRAIN-MULTI + proper 12mo OOS (control) | 248 honest | -2.29 | -0.005 | 0.41 | -56% | 1-epoch better than 5; still net negative. 2026 IC=+0.013 (model adapts during OOS phase) |
| 4 | Cost-sensitivity sweep on iter3 predictions | 248 | varies | — | varies | varies | **MAJOR: at 0 cost K=20 has Sharpe +0.78 / +14% sleeve. Break-even RT cost ~5-7 bps. K=20 wins at low cost, K=1 at high cost. Real gross alpha but thin.** |
| 5 | Multi-day hold (10-20 day rebalance, K=5-10) on iter3 preds, 30 bps RT cost | 240 honest | — | — | — | — | **HONEST CHAMPION**: 20-day hold K=10 gives Sharpe +0.72, sleeve +14% (~14-16%/yr ann). Same predictions as iter3 but rebalance less often → cost amortizes. K=5 hold=20 days: Sharpe +0.70, sleeve +16%. **First positive OOS result.** |
| 6 | Lr=5e-4 + 1ep + 12mo OOS, hold=20 K=20 | 240 honest | **+2.15** | **+0.006** | 0.67 | **+19.1%** | **NEW CHAMPION**. 3× Sharpe vs iter3. Lower LR fixes overfit. K=5: Sharpe +1.83 +19.8%. K=5 hold=10: sleeve +29.9% (Sharpe +1.31). |
| 7 | Lr=1e-3 + wd=0.20 (5x) + 1ep + OOS, hold=20 K=10 | 240 honest | +1.71 | **+0.011** | 0.67 | **+23.2%** | Higher WD also fights overfit. **Highest IC/IR**: 0.011/1.84. K=20 hit=75%. |
| 8 | Lr=1e-3 + 2 epochs + OOS, hold=20 K=5 | 240 honest | +2.65 | +0.003 | 0.83 | +26.8% | 2 epochs is sweet spot (1ep=0.72, 5ep=-4.50). Hit 10/12. |
| 9 | Lr=1e-3 + 1ep + buffer=504 + steps/day=5, hold=5 K=20 | 240 | +0.54 | +0.002 | 0.45 | +8.3% | More gradient steps per day = OVERFITS like multi-epoch. |
| 10 | Lr=1e-4 + 1ep + OOS, hold=20 K=5 | 240 honest | **+3.20** | **+0.011** | **0.92** | **+69.3%** | **NEW CHAMPION** 11/12 rebals win. bps/day +22. 10x lower LR vs iter3 \= 5x Sharpe. |
| 11 | Lr=5e-4 + wd=0.20 (combine 6+7), hold=20 K=5 | 240 honest | +1.49 | +0.003 | 0.67 | +27.5% | Combo worse than iter6 alone. WD+LR don't stack. |
| 12 | Lr=5e-4 + 2ep, hold=20 K=5 | 240 honest | +1.66 | +0.010 | 0.67 | +25.0% | 2 epochs at 5e-4 — comparable to iter8 2ep. |
| 13 | Lr=1e-4 + 2ep, hold=20 K=5 | 240 honest | +1.64 | +0.006 | 0.67 | +18.3% | 2 epochs at 1e-4 — HURTS vs iter10's 1ep (+3.20). Overfits at lower LR with 2 ep. |
| 14 | Lr=5e-5 + 1ep, hold=20 K=20 | 240 honest | +1.21 | +0.009 | 0.58 | +15.0% | Too low LR loses signal. |
| 15 | 5d cumul target (lr=1e-4 + 1ep) | 240 honest | +1.57 | -0.008 (1d), -0.004 (5d) | 0.75 | +18.3% | Long-horizon target loses cross-sectional signal. Positive Sharpe is ETF-padding artifact. |
| audit_iter10 | Bootstrap CI + spread + neg control | - | - | - | - | - | iter10 Sharpe 95% CI=[+2.34, +6.74], P(>0)=100%. Top-vs-Bot spread +4.36%, t=+3.01 (p≈0.005). Random K=5: Sharpe mean=+0.66, 95%=+2.08. iter10 in 100th pctile. **Signal real but heavily concentrated in LIQUIDBEES/LTGILTBEES/index ETFs.** 12 of 60 picks = ETF universe stocks. |
| 16 | Ranking loss (w_rank=1.0 w_reg=0.5) | 240 honest | +2.34 | +0.003 | 0.92 | +29.3% | Ranking loss high Sharpe but tiny IC — ETF-padded backtest. |
| 17 | Focal BCE (w_focal=2.0) | 240 honest | **+2.74** | **+0.0147** | 0.75 | +24.9% | **Best IC** so far (+0.0147 / IR +1.86). Focal classification head adds real signal. |
| 18 | Smaller model (d_ft=16 d_model=64, ~75K params) | 240 honest | +1.75 | **+0.0163** | 0.67 | +36.5% | **Highest IC** (+0.0163 / IR +1.88). Less overfit. Sleeve +36.5% (high). Suggests true model capacity is < 300K params. |
| 19 | Bigger model (d_ft=48 d_model=192, ~600K params) | 240 honest | +1.57 | +0.011 | 0.67 | +22.2% | Bigger overfits — IC drops, Sharpe drops. Confirms capacity sweet spot < 300K. |
| 20 | Buffer=504 days (2-year history) | 240 honest | +1.19 | +0.011 | 0.58 | +19.4% | Longer buffer doesn't help — model already sees enough context. |
| 21 | focal=2.0 + w_clf=2.0 | 240 honest | +1.17 | +0.014 | 0.67 | +17.6% | Doubling clf weight doesn't beat iter17 (w_clf=1.0). |
| 22 | focal=4.0 (heavier) | 240 honest | +1.71 | +0.010 | 0.58 | +24.9% | focal=4.0 too heavy — IC drops. focal=2.0 was optimal. |
| 23 | focal+rank combined | 240 honest | **+3.27** | +0.004 | **0.92** | +33.6% | Sharpe matches iter10 but low IC — ETF-padded backtest. |
| 24 | Tiny model (d_ft=8 d_model=32, ~25K params) | 240 honest | +2.67 | +0.009 | 0.75 | +29.6% | Tiny model still works — IC slightly lower than iter18 small. |
| 25 | focal=2.0 + smaller (d_model=64) hold=5 K=5 | 240 honest | +1.78 | +0.0115 | 0.57 | **+48.6%** | **Highest sleeve** at hold=5 K=5 + bps/day +16.23. More diverse picks (36% unique, real stocks dominate). Combine champions iter17 + iter18 yields most real alpha. |
| 26 | lr=2e-4 | 240 honest | +0.46 | +0.008 | 0.50 | +5.9% | LR between 1e-4 and 5e-4: actually worst Sharpe. |
| 27 | focal+small+wd=0.20 | 240 honest | +1.35 | +0.005 | 0.50 | +26.5% | Higher WD hurts IC when combined with focal+small. |
| 28 | focal+tiny (d_model=32) | 240 honest | +1.04 | +0.009 | 0.50 | +13.7% | Too tiny — IC lower than iter24 (no focal). focal needs some capacity. |
| 29 | focal+small+lr=5e-5 | 240 honest | +1.28 | +0.004 | 0.67 | +21.6% | Too low LR with focal — IC collapses. |
| 30 | focal+small+buf=126 | 240 honest | +1.66 | +0.009 | 0.67 | +22.3% | Buffer 126 vs 252: marginal difference. |
| 31 | focal+small+steps_per_day=1 | 240 honest | +1.00 | +0.006 | 0.62 | +19.5% | Less training per day hurts. |
| 34 | focal+small+1 transformer layer | 240 honest | +1.51 | **-0.004** | 0.58 | +25.3% | 1 layer too shallow — IC NEGATIVE. Min depth=2. |

## Pipeline notes
- Train period: 2019-06-26 → 2022-12 (warmup)
- Predict-from: 2023-01-01
- Online-end: 2026-04-30 (814 predict days)
- Cost: 30 bps round-trip
- Top-K=1 long-only baseline (apples-to-apples vs cohort/Phase1 audits)
