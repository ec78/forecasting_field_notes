# selective-hybrid-forecasting

A working repo for an ongoing Medium series on emerging time-series
forecasting methods: small, conceptual demonstrations built so that each
post's claims are checked against a shared, honest evaluation design rather
than just asserted.

## How this repo works

- **[`literature/`](literature/)** — candidate papers as they're found (see
  [literature/index.md](literature/index.md)).
- **[`topics/`](topics/)** — how a week's findings become a blog topic (see
  [topics/README.md](topics/README.md)); a scheduled weekly crawl feeds this
  automatically.
- **`notebooks/<NN>-<slug>/`** and **`results/<NN>-<slug>/`** — one numbered,
  matching pair per post.
- **[`posts/`](posts/)** — prose drafts, one `<NN>-<slug>.md` file per post
  (or `<NN>-<slug>/` with the same name if it needs supporting assets, e.g.
  a rendered equation image), written once a notebook's results are in. Not
  published from here — Eric publishes the finished draft to Medium; the
  file stays as a historical record afterward.
- **[`src/`](src/)** — the shared toolbox (data loaders, the rolling-origin
  evaluation harness, model wrappers) reused across posts.
- **[`CLAUDE.md`](CLAUDE.md)** — repo conventions and the writing-style guide
  used when drafting posts.

## Shared evaluation design

Every notebook, regardless of post, uses the same rolling-origin evaluation:

- Identical forecast origins and horizons across all models being compared.
- Metrics: MAE, RMSE, MASE (scaled by the in-sample naive/seasonal-naive
  error), and directional accuracy where it's meaningful.
- Runtime and any training/adaptation cost is recorded alongside accuracy —
  a model that wins by a fraction of a point of MASE but costs 50x the
  compute is a different story than one that wins for free.
- A seasonal-naive or naive baseline is always in the comparison.
- Any weight, scale, or hyperparameter that isn't fixed in advance is
  selected on a validation slice only, never on the test period.

The harness lives in [src/evaluation.py](src/evaluation.py); reuse it for
new posts rather than rebuilding it per-notebook.

## Posts

### 01 — Selective hybrid forecasting

*When is a pretrained model worth using on its own, versus as one ingredient
in something more selective?* Companion to two earlier posts:

- [A Non-Expert's Attempt at Tuning Transformer Hyperparameters](https://medium.com/@eclow/chasing-chronos-663cc8883558)
- [Transformers vs. ARIMA: Do More Complex Models Produce Better Forecasts?](https://medium.com/@eclow/transformers-vs-arima-do-more-complex-models-produce-better-forecasts-28ca372052d4)

Those posts asked whether more complex models beat simple ones. This series
asks a follow-up question: **when a pretrained model *is* worth using, is it
better used on its own, or as one ingredient in something more selective?**

Two series are used throughout (see [data/README.md](data/README.md)):

- **CISO hourly electricity demand** (2019-2026) — seasonal, high-signal.
- **SPY daily adjusted close** (1993-2026) — close to a random walk, used to
  show what happens when there isn't much signal to extract.

| Notebook | Question |
|---|---|
| [01_zero_shot_and_local_baselines.ipynb](notebooks/01-selective-hybrid-forecasting/01_zero_shot_and_local_baselines.ipynb) | When does a pretrained zero-shot forecast add value beyond local baselines, and does a validation-weighted hybrid do better than either alone? |
| [02_residual_memory_correction.ipynb](notebooks/01-selective-hybrid-forecasting/02_residual_memory_correction.ipynb) | Can retrieving residuals from similar past situations correct a frozen base forecast? |
| [03_frequency_informed_features.ipynb](notebooks/01-selective-hybrid-forecasting/03_frequency_informed_features.ipynb) | Does giving a flexible model explicit Fourier terms for known periodicity help it generalize? |

**1. Zero-shot is the starting line.** Compares seasonal-naive, a local
ARIMA baseline, Chronos-2 zero-shot, and a validation-weighted
ARIMA/Chronos combination, at short/medium/long horizons. The hybrid weight
is chosen only on validation data. The point isn't that the hybrid always
wins — it's that "pretrained" and "locally demonstrated value" are
different claims, and this makes the gap between them visible.

**2. Residual memory as a correction layer.** Fits ARIMA and, at each
forecast origin, looks up the most similar prior lookback windows
(nearest-neighbor on a normalized window), retrieves the residual paths
that followed those windows, and adds a distance-weighted,
validation-scaled correction to the frozen base forecast. Causal retrieval
is enforced everywhere: no matched window or its residual path is allowed
to extend past the forecast origin (see `tests/test_residual_memory.py`).

**3. Classical structure as model input.** On hourly electricity demand,
compares a lag-only regression/small neural net against the same model
given Fourier terms for daily and weekly periodicity, alongside a SARIMAX
benchmark using those same Fourier terms as exogenous regressors. Not a
reproduction of CoSPOT — a small illustration of its underlying idea: a
flexible learner can benefit when classical analysis hands it a credible
representation of recurring structure.

References:

- Pendyala, V., Fu, Y., Yan, W., & Virani, N. (2026). *Assessing
  Covariate-Informed Grid Load Forecasting with a Time-Series Foundation
  Model.*
- He, Y., Cang, Y., Ning, Z., Wang, N., & Shi, L. (2026). *RATL: Learning
  from Retrieved Residuals for Robust Multivariate Time-Series Forecasting.*
- Choi, S., Kim, H., Lee, J.-G., & Park, C. (2026). *Compositional Spectral
  Prompts for LLM-based Online Time Series Forecasting.*
- Chen, L., & Wang, W. (2026). *From Vector Autoregressions to AI-based Time
  Series Forecasting: A Review.*
- Garcia, C., & Remy, G. (2026). *Statistical Benchmarking of Transformer
  Models in Low Signal-to-Noise Time-Series Forecasting.*

### 02 — Excitation: does a good forecast mean the model knows the lever?

Published: [Can Your Forecast Predict the Effect of a Decision?](https://medium.com/@eclow/can-your-forecast-predict-the-effect-of-a-decision-7205e481a0d9)

*If a forecast closely matches what a system actually did, does that mean the model has
learned how the system responds to its controls — or just what its existing policy tends
to do?* Inspired by Amria et al. (2026), *Time-series Foundation Models for Predictive
Control: The Role of Excitation* (`literature/excitation-2026.pdf`), who find that a
TSFM can forecast a building's temperature accurately under its existing controller while
being nearly insensitive to a hypothetical change in that controller's setpoint — unless
the context it conditions on contains genuinely independent variation in the control
signal ("excitation").

This is the one demo in the series built on a synthetic system rather than real data —
see the notebook for why: testing whether a forecast responds *correctly* to an
intervention needs a ground-truth counterfactual that no historical series can provide.

| Notebook | Question |
|---|---|
| [01_excitation_and_control_fidelity.ipynb](notebooks/02-excitation-for-control/01_excitation_and_control_fidelity.ipynb) | Does Chronos-2's forecast respond correctly to a change in a control input, and does that depend on whether its context contains independent variation in that control — separately from whether its forecast of the unperturbed trajectory is accurate? |

A small linear "toy thermostat" system is driven two ways to build historical context —
an **unexcited** closed-loop policy (control is a deterministic function of state and
time) and an **excited** APRBS signal (control varies independently of state) — then
scored with the paper's *response gain* metric: given several matched counterfactual
probes that share context and differ only in future control, does the predicted response
align with the true one? Chronos-2 (given control as a known-future covariate) and a
linear ARX baseline fit on the context are compared across context lengths and excitation
levels.

References:

- Amria, M., Hoffmann, J., Bordne, P., Rothenhäuser, A., Frison, L., Walnum, H. T.,
  Gros, S., & Bödecker, J. (2026). *Time-series Foundation Models for Predictive Control:
  The Role of Excitation.* NeurIPS 2026 Workshop: TS-LIMITS.

### 03 — Is the expensive refiner worth it?

*Does predicting a cheap, global shape first — then spending an expensive model's effort
only on what's left over — actually beat running the expensive model on its own?*
Inspired by SCALER (Le et al., 2026, KDD — `literature/scaler-2026.pdf`), a fully
trained coarse-Transformer + frozen-LLM-refiner architecture; not reproduced here, just
its underlying idea, rebuilt as a small analogue from pieces this repo already has:
seasonal-naive as the coarse shape (Post 01), the nearest-neighbor residual-memory
retrieval from Post 01 Demo 2 as a cheap refiner, and Chronos-2 zero-shot as an expensive
refiner — both refiners pointed at the coarse model's residual instead of the raw series.

| Notebook | Question |
|---|---|
| [01_coarse_to_fine_hybrids.ipynb](notebooks/03-coarse-to-fine-hybrids/01_coarse_to_fine_hybrids.ipynb) | Does a cheap coarse shape plus a refiner beat Chronos-2 forecasting the raw series directly — on accuracy, on cost, or either? |

**Honest result, at full scale (n=56 origins per horizon, not a smoke-test sample):
no.** Neither refiner, tuned or not, beats Chronos-2 on the raw series at the short or
medium horizon; at the long horizon the best any tuned variant achieves is matching plain
seasonal-naive, because validation data selects a refinement weight of zero there — it
correctly learns there's nothing useful to add, not that the mechanism works. A likely
reason: Chronos-2 already captures CISO's daily/weekly shape well enough zero-shot that
handing it the shape for free has nothing left to buy, unlike SCALER's own setting, where
the refiner is an LLM without native time-series pretraining. A second, unplanned finding:
the "cheap" residual-memory refiner isn't actually cheap at full history scale — its
brute-force nearest-neighbor search over tens of thousands of candidate windows ends up
costing as much as a Chronos-2 call, sometimes more.

References:

- Le, X.-M., Tran, M.-T., Luo, L., Aickelin, U., Phung, D., & Le, T. (2026). *Efficient
  Test-Time Scaling for LLM-based Time Series Forecasting.* KDD 2026.

### 04 — Does an online corrector earn its place once you wait for the labels?

*If a small model watches a frozen forecaster's errors as they arrive and corrects its next
forecast, does that beat the frozen model — once you respect that an h-step-ahead forecast's
error only exists h steps later?* Inspired by two papers that pull in opposite directions:
ORCA (Dai et al., NeurIPS 2026), an online residual adapter for frozen, black-box foundation
models that reports ~6% average gains, and Training on the Future (Fentazi, Ameur & Ksentini,
2026), an audit finding that under causal label delay most published test-time adapters
gain little and a closed-form RLS-Bank sets a bar they rarely clear. Neither tests ORCA
against that bar; this does, in miniature, correcting Chronos-2's *own* errors on CISO and
SPY — an ORCA-style closed-form ridge adapter (with and without its error-gated router), the
RLS-Bank rebuilt from the paper, and Post 01's residual memory run online — with label delay
(leaky reference / minimum causal / late) as an explicit axis.

| Notebook | Question |
|---|---|
| [01_online_residual_correction.ipynb](notebooks/04-online-residual-correction/01_online_residual_correction.ipynb) | Does an online residual corrector on frozen Chronos-2 beat the frozen model under causal label delay — and does ORCA's adapter beat a no-tuning closed-form RLS-Bank? |

**Result:** *(full-scale run in progress — filled in from
`results/04-online-residual-correction/metrics/*_fullscale.csv`)*

References:

- Dai, X., Liu, Y., Xia, H., Hu, Y., Dong, Z., Yang, J.-M., & Xu, Q. (2026). *Learning the
  Context of Errors: Black-Box Online Adaptation of Time Series Foundation Models.*
  NeurIPS 2026. arXiv:2606.14222.
- Fentazi, M. R., Ameur, M., & Ksentini, A. (2026). *Training on the Future: A Delay-Aware
  Audit of Test-Time Adaptation for Time-Series Forecasting.* arXiv:2610.12232.

*(Future posts get their own `### NN — Title` section here, following the
same shape: question, notebook table, one paragraph per demo, references.)*

## Repository structure

```
selective-hybrid-forecasting/
├── literature/                candidate papers — see literature/index.md
├── topics/
│   ├── README.md              how a week becomes a blog topic
│   ├── backlog.md             de-duplicated candidate topics
│   ├── digests/                dated weekly crawl output (YYYY-MM-DD.md)
│   └── chatgpt-reports/        dated raw drops from the external ChatGPT task
├── data/
│   ├── README.md
│   └── raw/                   electricity_ciso.csv, spy.csv (+ metadata)
├── posts/                      prose drafts, one <NN>-<slug>.md per post
├── notebooks/
│   ├── 01-selective-hybrid-forecasting/   one subfolder per post
│   ├── 02-excitation-for-control/
│   ├── 03-coarse-to-fine-hybrids/
│   └── 04-online-residual-correction/
├── src/
│   ├── data.py                 loaders + train/val/test split
│   ├── evaluation.py            metrics + rolling-origin harness
│   ├── models.py                 naive / ARIMA / Chronos / hybrid / Fourier+SARIMAX
│   ├── residual_memory.py      causal nearest-neighbor residual retrieval
│   ├── control.py               synthetic controllable system + response-gain metric
│   ├── coarse_to_fine.py        cheap shape model + pluggable residual refiner
│   ├── online_correction.py     delay-aware replay + online residual correctors
│   └── dataimport.py            optional: refresh the raw CSVs
├── tests/                     unit tests for the metrics and causal retrieval logic
├── results/
│   ├── 01-selective-hybrid-forecasting/
│   │   ├── figures/
│   │   └── metrics/
│   ├── 02-excitation-for-control/
│   │   ├── figures/
│   │   └── metrics/
│   ├── 03-coarse-to-fine-hybrids/
│   │   ├── figures/
│   │   └── metrics/
│   └── 04-online-residual-correction/
│       ├── figures/
│       └── metrics/
└── CLAUDE.md                  repo conventions + writing-style guide
```

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate   # source .venv/bin/activate on macOS/Linux

pip install -e ".[notebooks,dev]"   # base install + Jupyter + pytest
pip install -e ".[chronos]"          # adds torch + chronos-forecasting (large download)
```

Run the test suite with `pytest`. Chronos-2 (`amazon/chronos-2`) downloads
from Hugging Face on first use and runs on CPU unless a CUDA GPU is
available; expect zero-shot inference to be the slowest part of Post 01's
Demo 1, the Chronos branch of Demo 2, all of Post 02, and the Chronos-refiner
branch of Post 03, and the base stream of Post 04 (one Chronos-2 call per
time step — the slowest notebook in the repo at full scale).
