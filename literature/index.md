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
| [ctrl-2026.pdf](ctrl-2026.pdf) | CTRL: Control-Based Time Series Forecasting with LLM-Guided Residual Learning | Kim, Ji, Lee, Kim, Jang — Yonsei University | 2026-09-28 | skipped | Full paper read 2026-10-30. LLM agents emit one control vector shared across all samples (3 LLM calls) that conditions a trained ~400K-param residual decoder on STL components; LLM's measured contribution is small (two ETT datasets) and foundation-model backbones are untested. Parked: needs an LLM API for a small effect. arXiv 2609.23257. |
| [excitation-2026.pdf](excitation-2026.pdf) | Time-series Foundation Models for Predictive Control: The Role of Excitation | Amria, Hoffmann, Bordne, Rothenhäusler, Frison, Walnum, Gros, Bödecker — U. Freiburg / NTNU | 2026-10-15 | used-in-02-excitation-for-control | Full paper read. Core mechanism (response gain, excited vs. unexcited context) reproduced on a small synthetic system in `notebooks/02-excitation-for-control/`. |
| [arXiv 2607.02623](https://arxiv.org/abs/2607.02623) | Evaluating Time Series Foundation Models for Electricity Price Forecasting: Contamination Risk, Distributional Shifts, and Covariate Dependence | Pan & Ezzat — ICML 2026 workshop | 2026-10-16 | skipped | Full paper read 2026-10-30. Contamination mitigation = evaluate on a period after pretraining; Post 01's CISO test window already postdates Chronos-2, and SPY had no zero-shot win to deflate, so nothing to re-check. Covariate finding (electricity *price*) overlaps the t0 topic. |
| [arXiv 2605.05218](https://arxiv.org/abs/2605.05218) | Horizon-Constrained Rashomon Sets for Chaotic Forecasting | arXiv 2026 | 2026-10-16 | unread | Predictive multiplicity across horizons: tied short-horizon accuracy, diverging long-horizon paths. |
| [arXiv 2610.09994](https://arxiv.org/abs/2610.09994) | Temporal Predictive Multiplicity: Equally Accurate Time Series Models Yield Different Forecast Trajectories | Albini, Toni, Mishra, Leofante — arXiv Oct 2026 | 2026-10-30 | reading | Full paper read 2026-10-30. Pointwise / horizon-wise / trajectory-level multiplicity hierarchy with bounds; 19 neural architectures x 11 datasets. Backs the predictive-multiplicity backlog row (runner-up). |
| [arXiv 2610.12232](https://arxiv.org/abs/2610.12232) | Training on the Future: A Delay-Aware Audit of Test-Time Adaptation for Time-Series Forecasting | Fentazi, Ameur & Ksentini — EURECOM, arXiv Oct 2026 | 2026-10-16 | used-in-04-online-residual-correction | Full paper read 2026-10-30. Delay-parameterized audit of TTA (label of origin s usable only at s+d, d>=H); closed-form RLS-Bank (6 forgetting factors + passthrough, per-coordinate median) matches or beats most published adapters; leaky updates inflate gains up to 110%. RLS-Bank re-implemented from the paper's Appendix D (repo has no license file). |
| [arxiv 2608.14106](https://arxiv.org/abs/2608.14106) | Forecast Collapse in Time-Series Foundation Models | Wan, Ma et al. — arXiv, Aug 2026 | 2026-10-19 | unread | TSFMs give near-flat, poorly-ranked stock return forecasts; relevant to SPY series. Abstract not verified directly. |
| [arxiv 2609.24559](https://arxiv.org/abs/2609.24559) | t0: A Time-Series Foundation Model for Forecasting with Context | The Forecasting Company — arXiv, Sep 2026 | 2026-10-19 | unread | Open-weights TSFM using past and known-future covariates; pairs with Demo 3 / CISO. |
| [scaler-2026.pdf](scaler-2026.pdf) | Efficient Test-Time Scaling for LLM-based Time Series Forecasting (SCALER) | Le, Tran, Luo, Aickelin, Phung, Le — KDD 2026 | 2026-10-19 | used-in-03-coarse-to-fine-hybrids | Full paper read. SCALER is a fully trained coarse-Transformer + frozen-LLM-refiner architecture (multi-scale token refinement, custom loss) — not reproduced. Its idea (cheap global shape first, expensive model only refines what's left) rebuilt as a small analogue from existing repo pieces in `notebooks/03-coarse-to-fine-hybrids/`. |
| [arxiv 2606.14222](https://arxiv.org/abs/2606.14222) | Learning the Context of Errors: Black-Box Online Adaptation of Time Series Foundation Models (ORCA) | Dai et al. — Ant International / CUHK, NeurIPS 2026 | 2026-10-26 | used-in-04-online-residual-correction | Full paper read 2026-10-30 (and released code checked: training snapshots and router errors only use matured labels). Linear adapter on [lookback, base forecast] -> residual, buffer refit every H, EMA-error Boltzmann router anchored to the frozen model. Rebuilt as a closed-form ridge analogue (same linear hypothesis class) in `notebooks/04-online-residual-correction/`. Code Apache-2.0. |

**Status values:** `unread` / `reading` / `used-in-<post-slug>` / `skipped`
(skipped = read, not a fit — leave a one-line reason in Notes rather than
deleting the row).
