# Preregistration — Post 05: Declare the null before you look

**Status: FROZEN 2026-11-18.** The freeze is this file's first commit on
`main`, as pushed to GitHub (`git log --follow -- notebooks/05-declare-the-null/PREREGISTRATION.md`).
The GitHub push time is the third-party timestamp, and local commit dates don't
count. From here on, changes go in §10 as dated amendments, and nothing in
§§1–9 is edited in place. The only exception is filling in the blank SHAs and
dates in §9 as each step happens.

Modelled on Upadhyay (2026), *Do Time-Series Foundation Models Forecast
Index-Futures Returns? A Preregistered Evaluation* (arXiv:2611.12141). That
study is intraday ES/NQ futures. This one is a small daily-SPY analogue, using
the repo's shared rolling-origin harness (`src/evaluation.py`).

---

## 1. Question

> On daily SPY, does frozen, zero-shot Chronos-2 forecast h-day log returns
> better than a no-change forecast, once the loss, the family of comparisons
> and the multiplicity correction are fixed before the data is seen?

The hook, which is already known and not part of the confirmatory test: in
Post 04 (`results/04-online-residual-correction/metrics/online_correction_vs_frozen_fullscale.csv`),
frozen Chronos-2 had about 12% lower scaled error than naive at h = 20 on
SPY. Its pairwise block-bootstrap interval excluded zero. That was one of 30
uncorrected comparisons in that table.

## 2. Data and windows

All windows are fixed here. The "seen" boundaries are taken from Post 04 so
that its h = 20 result can be re-examined on exactly the same origins.

| Window | Origins (date of last observed close) | Role | Status |
|---|---|---|---|
| Fit | through 2022-07-22 | fit the ridge comparator | seen |
| Validation | 2022-07-25 → 2024-07-19 | ridge α; nothing else is tuned | seen (Post 04 val) |
| **Seen** | 2024-07-22 → 2026-07-20 (500 origins) | exploratory replication of Post 04, including h = 20 | **seen** — Posts 01 and 04 scored forecasts here |
| **Blind** | 2026-08-17 → open date minus h | **confirmatory** | **unseen** — no SPY close after 2026-08-17 is in the repo |

- **Series.** SPY adjusted close from Yahoo Finance. `data/raw/spy.csv` ends
  2026-08-17 and is **not overwritten**, because Posts 01 and 04 depend on it.
  The blind stage uses a new file, `data/raw/spy_blind_<fetch-date>.csv`.
  That needs a small `--out` option on `src/dataimport.py`, added before the
  freeze.
- **Overlap check.** Over the overlapping dates, daily log returns in the new
  file must match `spy.csv` to within 1e-4. Re-adjustment for new dividends
  rescales the level but should not change returns. If the check fails, it is
  reported, and the blind stage still uses the new file.
- **Fetch hygiene.** The fetch prints only the row count and the first and last
  dates. No returns, prices or summary statistics are viewed before scoring.
- **Blind sample size.** If opened around 2026-11-20, this is about 67 trading
  days, giving roughly 60–65 usable origins at h = 1 and h = 5, and about 47 at
  h = 20.
- **Contamination.** Chronos-2's weights were released in Oct 2025, so the
  blind window postdates them entirely. The seen window is partly
  pre-release (2024-07 → 2025-10), which is a further reason it is
  exploratory only.

## 3. Target, loss and the smallest meaningful improvement

- **Target.** `y = log(P[t+h] / P[t])`, where P is the adjusted close and
  h ∈ {1, 5, 20} trading days. Every forecaster is scored against this one
  column.
- **Loss.** Absolute error on the return, `|y − f|`.
  - The headline number is the **MAE ratio against no-change**:
    `Σ|y − f| / Σ|y|`. Below 1 favours the model.
  - Running the harness on log price gives exactly this, because
    `|log P[t+h] − log P̂[t+h]| = |y − f|`.
  - MASE is also reported, for continuity with Posts 01–04.
- **Smallest meaningful improvement (SMI).** A cell "beats no-change" only if
  both conditions hold:
  - its MAE ratio is **≤ 0.995**, and
  - the upper bound of its 95% paired moving-block bootstrap interval is
    **< 1**.

  The bootstrap uses `src.evaluation.paired_block_bootstrap` on `|y − f| − |y|`,
  with block length `max(5, h)`, `n_boot = 2000` and `seed = 20261118`.

## 4. Forecasters

All Chronos-2 runs use `amazon/chronos-2`, frozen and zero-shot, with context
`max_context_length = 512` (as in Post 04). The point forecast is the median
quantile.

**Confirmatory family**

| Name | Input | Decoding back to `f` |
|---|---|---|
| Chronos-2 · R0 | adjusted-close price levels | `log(median P̂[t+h]) − log P[t]` (Post 04's setup) |
| Chronos-2 · R1 | daily log returns | sum of the per-step median returns over h steps (as in the paper) |
| Chronos-2 · R2 | log returns ÷ causal EWMA volatility (λ = 0.94, RiskMetrics daily) | `σ[t+1\|t] × Σ` per-step medians, with σ held fixed over the horizon |

**Benchmark**

- **No-change:** `f = 0`.

**Comparators.** These are outside the family and never enter the SPA test.

| Name | Definition | Question it answers |
|---|---|---|
| Drift | `f = h × mean(daily log return over the same 512-day context Chronos-2 sees)` | Ingredient control: is a Chronos-2 "win" more than an upward drift the series already contains? |
| Ridge | per-horizon ridge on the last 20 log returns plus their 5- and 20-day sums; fit on the Fit window, α from {0.1, 1, 10, 100, 1e3, 1e4} by validation MAE, then frozen; refit once on everything through 2026-08-17 before the blind stage | Does Chronos-2 add anything over a cheap model with the same information? |
| Always-up | `sign(f) = +1` | Base rate for directional accuracy only |

## 5. Confirmatory test (blind window only)

- **Family C:** {R0, R1, R2} × {h = 1, h = 5} = **6 cells**.
- **Why h = 20 is excluded.** About 47 heavily overlapping origins amount to
  roughly 2–3 independent 20-day periods. That is not enough to test anything,
  so h = 20 is scored on the blind window **descriptively only**.
- **Test.** Hansen SPA, H0: "no cell in C beats no-change."
  - Implementation: `arch.bootstrap.SPA`, with the benchmark loss `|y|` and the
    model losses `|y − f|` (a T × 6 matrix). Origins are those where the h = 5
    target exists (the common sample).
  - Settings: stationary bootstrap, mean block length 5, 1,000 reps,
    studentized, `seed = 20261118`. The **consistent** p-value is the one
    reported.
  - A unit test pins down the sign convention before the freeze, using a model
    built to have lower loss than the benchmark.
- **Decision rule.** The verdict is "Chronos-2 shows skill over no-change on
  daily SPY" only if **both** hold:
  1. SPA consistent p < 0.05, **and**
  2. at least one cell in C meets the SMI.

  Otherwise the verdict is **"no skill detected; edges smaller than the §7
  detection limit are not excluded."**

## 6. Registered descriptive analyses (reported whatever they show, never claimed as findings)

- **D1 — The multiplicity lesson.**
  - Pairwise SMI verdicts for every cell in the 9-cell set (R0/R1/R2 ×
    h = 1/5/20), on both windows.
  - The number of pairwise "wins" next to the SPA verdict for the same set.
  - On the seen window, SPA is also run over all 9 cells (mean block 20),
    labelled exploratory.
- **D2 — Replicating Post 04's h = 20 result** on the seen window: first Post
  04's own scaled-error comparison, then the MAE ratio, then against **drift**,
  then inside SPA.
- **D3 — Price-level correlation.** `corr(P̂[t+h], P[t+h])` for every cell and
  for no-change, to show how "tracking the price" flatters every forecaster.
- **D4 — Directional accuracy against the always-up base rate,** not against
  50%, with a binomial test against that base rate.
- **D5 — Dispersion ratio.** `sd(f) / sd(y)` per cell, as a flatness check
  borrowed from Forecast Collapse (Wan et al. 2026). It is descriptive only.
- **D6 — Comparators.** Paired intervals for Chronos-2 against drift and
  against ridge.

## 7. Detection limits (run on seen data and committed **before** the blind window is opened)

- **Planted-signal series.**
  - Take SPY's real |daily log return| sequence from the validation window.
  - Replace the signs with a Markov chain that repeats the previous sign with
    probability 0.5 + δ, for δ ∈ {0.05, 0.1, 0.2, 0.3}, with 20 realisations
    each.
  - Score windows matching the blind size (65 origins) and the seen size (500).
- **Detectors.** Each detector counts as detecting the signal when it meets the
  SMI:
  - **Oracle:** `f = Σ_{k=1..h} (2δ)^k · s_t · m`, where `s_t` is the current
    sign and `m` is the mean |r| over the trailing 512 days.
  - **Ridge**, as defined in §4.
  - **Chronos-2 · R1** at blind size only, to bound runtime.
- **Reporting.** Detection rate against δ, with 95% Wilson intervals. The
  **detection limit** is the smallest δ at which the oracle detects in at
  least 80% of realisations. Smaller edges are stated as not excluded.

## 8. Predictions (written before any analysis; confirmed by Eric 2026-11-18)

- P1: No cell in C meets the SMI on the blind window, and SPA does not
  reject.
- P2: The detection limit at blind size is large: the oracle needs δ ≥ 0.2
  or more. A null therefore says little about realistic edges.
- P3: On the seen window, at least one of the 9 cells "wins" pairwise, but the
  9-cell SPA does not reject.
- P4: Post 04's h = 20 advantage over naive mostly disappears against drift,
  because 2024-07 → 2026-07 was an up-market. This was *not* checked
  beforehand; drift was proposed before anyone looked.
- P5: Every cell, and no-change, has price-level correlation > 0.99, and
  no-change is highest or tied.

## 9. Procedure and chronology

1. **Freeze.** Commit this file and push: 2026-11-18. The SHA is this file's
   first commit (see the status line at the top).
2. **Build and code freeze.**
   - Build `01_declare_the_null.ipynb` with `SMALL_SCALE` and the `src/` additions
     (decoders, the SPA wrapper with its sign test, the `dataimport --out`
     option).
   - Run the seen stage and the §7 detection limits in full.
   - Commit and push. SHA: `________`.
3. **Open the blind window once.**
   - Fetch to the new file (see §2), run the blind stage once, and commit the
     raw outputs unedited. SHA: `________`.
   - **Open date (decided 2026-11-18): as soon as step 2's code freeze is
     pushed.** The blind window is every SPY close after 2026-08-17 available
     at fetch time, expected to be about 65 trading days. The open date was
     chosen over a later date with about 125 days, to keep the post timely.
     The small sample is accepted, and §7 states what it can and can't detect.
   - The window opens **once**. It is not reopened or extended later. Doing so
     would be peeking at an interim result and would undo the point of the
     post.
4. **Write up.** Any analysis not listed above is labelled post hoc wherever it
   appears.

## 10. Amendments

*(Dated entries only, written after the freeze. Each entry says whether the
blind window had been opened at the time.)*
