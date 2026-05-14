---
name: Curriculum training for TradeableHead
description: User prefers two-stage training — long history with Phase 3 encoder first, then OB fine-tune — over single-stage on the OB-rich window
type: feedback
originSessionId: 03a1ea77-d596-424a-a7fc-412d4a5752b8
---
For TradeableHead (and likely similar heads later): train Stage 1 on the full multi-year history (e.g. 2019→2026-02) with the text-only Phase 3 encoder, then fine-tune Stage 2 on the recent OB-rich period with the Phase 4 OB-conditioned encoder.

**Why:** The OB-rich window is only a few months; training a head only on that window risks overfitting to a single regime. Long-history Stage 1 lets the head learn the general "what predicts a profitable LONG/SHORT setup" pattern across many regimes; Stage 2 then specializes to OB-conditioned signal. User asked for this explicitly when offered a single-stage sweep.

**How to apply:** When the user asks to train a new head on a recent fine-grained signal (OB, news, etc.), default to: (1) propose Stage 1 on the long-history label set with the broadest available encoder, (2) Stage 2 fine-tune from that checkpoint on the narrow window with the richer encoder. Don't jump straight to the narrow window.
