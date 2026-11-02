# Literature index

A running backlog of papers worth reading — candidates for future demos, not
necessarily things already used. For papers actually cited by a post, see
that post's section in the root [README.md](../README.md) instead; this
index is the superset.

Drop PDFs directly in this folder as you find them (`slug-year.pdf`), then
add a row here. The weekly digest routine (see
[topics/README.md](../topics/README.md)) also adds rows here for promising
papers it finds, without downloading the PDF — those rows link out instead
of naming a local file.

| File / Link | Title | Authors / Venue | Added | Status | Notes |
|---|---|---|---|---|---|
| [ctrl-2026.pdf](ctrl-2026.pdf) | CTRL: Control-Based Time Series Forecasting with LLM-Guided Residual Learning | Kim, Ji, Lee, Kim, Jang — Yonsei University | 2026-09-28 | unread | LLM-guided residual correction — close cousin of Demo 2's residual-memory mechanism; worth comparing approaches. |
| [excitation-2026.pdf](excitation-2026.pdf) | Time-series Foundation Models for Predictive Control: The Role of Excitation | Amria, Hoffmann, Bordne, Rothenhäusler, Frison, Walnum, Gros, Bödecker — U. Freiburg / NTNU | 2026-10-15 | used-in-02-excitation-for-control | Full paper read. Core mechanism (response gain, excited vs. unexcited context) reproduced on a small synthetic system in `notebooks/02-excitation-for-control/`. |
| [arXiv 2607.02623](https://arxiv.org/abs/2607.02623) | Evaluating Time Series Foundation Models for Electricity Price Forecasting: Contamination Risk, Distributional Shifts, and Covariate Dependence | Pan & Ezzat — ICML 2026 workshop | 2026-10-16 | unread | TSFMs depend on covariate support and don't reliably beat domain-specific methods; combination looks promising. Relates to Demo 1. |
| [arXiv 2605.05218](https://arxiv.org/abs/2605.05218) | Horizon-Constrained Rashomon Sets for Chaotic Forecasting | arXiv 2026 | 2026-10-16 | unread | Predictive multiplicity across horizons: tied short-horizon accuracy, diverging long-horizon paths. |
| [arXiv 2610.12232](https://arxiv.org/abs/2610.12232) | Training on the Future: A Delay-Aware Audit of Test-Time Adaptation for Time-Series Forecasting | arXiv Oct 2026 | 2026-10-16 | unread | Found via listing; abstract not checked (arxiv.org unreachable). Possible leakage audit of TTA methods. |
| [arxiv 2608.14106](https://arxiv.org/abs/2608.14106) | Forecast Collapse in Time-Series Foundation Models | Wan, Ma et al. — arXiv, Aug 2026 | 2026-10-19 | unread | TSFMs give near-flat, poorly-ranked stock return forecasts; relevant to SPY series. Abstract not verified directly. |
| [arxiv 2609.24559](https://arxiv.org/abs/2609.24559) | t0: A Time-Series Foundation Model for Forecasting with Context | The Forecasting Company — arXiv, Sep 2026 | 2026-10-19 | unread | Open-weights TSFM using past and known-future covariates; pairs with Demo 3 / CISO. |
| [scaler-2026.pdf](scaler-2026.pdf) | Efficient Test-Time Scaling for LLM-based Time Series Forecasting (SCALER) | Le, Tran, Luo, Aickelin, Phung, Le — KDD 2026 | 2026-10-19 | used-in-03-coarse-to-fine-hybrids | Full paper read. SCALER is a fully trained coarse-Transformer + frozen-LLM-refiner architecture (multi-scale token refinement, custom loss) — not reproduced. Its idea (cheap global shape first, expensive model only refines what's left) rebuilt as a small analogue from existing repo pieces in `notebooks/03-coarse-to-fine-hybrids/`. |
| [arxiv 2606.14222](https://arxiv.org/abs/2606.14222) | Learning the Context of Errors: Black-Box Online Adaptation of Time Series Foundation Models (ORCA) | Dai et al. — Ant International / CUHK, Jun 2026 | 2026-10-26 | unread | Online residual adapter for black-box TSFMs; cousin of Demo 2 and CTRL. Abstract seen via search summary only. |
| [arXiv 2610.16433](https://arxiv.org/abs/2610.16433) | ECHO: Correcting Time Series Foundation Model Forecasts with Historical Analogues and Errors | Zhang et al. — arXiv, Oct 2026 | 2026-11-02 | unread | Black-box TSFM correction via analogues + past errors; direct peer of Demo 2 and ORCA. Seen via search summary only. |
| [arXiv 2608.08825](https://arxiv.org/abs/2608.08825) | Hybrid neural-classical correction of a frozen TimesFM for high-frequency stock prediction | arXiv Aug 2026 (title uncertain; via RePEc listing) | 2026-11-02 | unread | Reportedly RF residual learning is the largest ablation gain; relevant to SPY. Verify title/abstract. |

**Status values:** `unread` / `reading` / `used-in-<post-slug>` / `skipped`
(skipped = read, not a fit — leave a one-line reason in Notes rather than
deleting the row).
