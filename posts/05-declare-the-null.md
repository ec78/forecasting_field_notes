<!--
OUTLINE — not a draft yet. Status tracked in topics/backlog.md
(in-progress, post: declare-the-null).

Prose gets written only after the blind window has been opened once, per
notebooks/05-declare-the-null/PREREGISTRATION.md. Every [RESULT] slot below is
filled from that run's committed outputs. Anything not listed in the
preregistration is labelled post hoc in the text.

Working titles:
  - Declare the Null Before You Look
  - What Does "Nothing" Look Like? Testing Chronos-2 on the S&P 500 the Honest Way
  - Did My Forecast Just Beat the Market? (No. Here's How to Check.)
Recurring phrase: "what does nothing look like?" (pairs with the series' "earn its place").
-->

# Declare the Null Before You Look

## Hook — "Did Chronos-2 just beat the market?"

- Open on Post 04's leftover: frozen Chronos-2 beat a no-change forecast on SPY
  at 20 days by about 12%, and the bootstrap interval excluded zero. *Taken at
  face value, that's a pretrained model forecasting the S&P 500 a month out.*
- Set up the doubt with a rhetorical question: is that skill, or is it what
  luck looks like after enough comparisons?
- Introduce the paper that prompted this (Upadhyay 2026): five foundation
  models, 160 combinations, intraday index futures, and **zero** beat
  no-change once the test was fixed in advance, even though several standard
  metrics said otherwise.
- State the promise: this post doesn't build a better model. It builds a
  better *question*, and then asks it of data nobody has seen yet.

## Why "no change" is so hard to beat (intuition 1)

- Use the everyday framing: if prices wander randomly, the best guess for
  tomorrow is today. Any forecast that moves away from that adds noise.
- Contrastive line: *a flat forecast isn't the model failing; it's the honest
  answer.* Tie this to Forecast Collapse (Wan et al.).
- Show the [RESULT] D5 dispersion ratio: how much Chronos-2's forecasts move
  compared with how much SPY actually moves.

## Three ways a forecast can look skilful without being skilful

Short, parallel subsections, each ending on a punchy line.

**1. Score the level and everyone's a genius (intuition 2).**
- Use the weather analogy ("same as today").
- [RESULT] D3: price-level correlation above 0.99 for everything, with
  no-change highest or tied.
- Lesson: score the change, not the level.

**2. Up days don't count as insight (intuition 4).**
- SPY rose on more days than it fell, so 50% is the wrong benchmark for
  directional accuracy.
- [RESULT] D4: directional accuracy against the always-up base rate.
- Mention the paper's sign-flip null in a sentence: its trivial
  revert-to-mean rule "ranked" moves at +0.35, and random signs reproduced
  95% of that.
- Lesson: *always ask what the number would be if there were no signal.*

**3. Drift is free (an ingredient control).**
- Over 2024–2026 the market mostly went up. A forecast that leans up wins
  without knowing anything.
- [RESULT] D2: Post 04's h = 20 win against drift, using the same 512 days of
  context.
- Callback: the paper's volatility model beat HAR but not the EWMA already in
  its own input. *Compare against the simplest thing that already holds the
  information.*

## More tickets, more winners (intuitions 3 and 5)

- Use the fund-manager analogy: pick the best of 100 after the fact and
  someone always looks brilliant.
- Count the tickets: 3 inputs × 3 horizons = 9 cells here. Post 04 ran 30
  comparisons.
- [RESULT] D1: pairwise "wins" against what SPA says for the same set, on the
  seen window.
- Overlap gets a sidebar: 20-day forecasts made on consecutive days share 19
  of their days, so 500 origins are worth far fewer independent tests. That
  explains block bootstraps, and why h = 20 is the most fragile result.
- Explain SPA in one plain sentence: *is the best of these better than doing
  nothing, after paying for the fact that we picked the best?*

## Calling the shot (intuition 6)

- The seen window isn't a fair test any more. Posts 01 and 04 already looked
  at it, and Chronos-2 may even have trained on part of it.
- The fix is to write the plan down first: models, horizons, loss, the
  threshold, the test and predictions. Commit and push it, then wait for data
  that didn't exist yet.
- Use the pool analogy ("call your shot"). The GitHub timestamp is the cheap,
  checkable version for a solo practitioner.
- Link to PREREGISTRATION.md and quote the five predictions verbatim, *before*
  showing results.

## What could this test even see? (intuition 7)

- Plant a fake edge in real SPY volatility and check whether a perfect
  forecaster (the oracle) could detect it at this sample size.
- [RESULT] §7 detection curve (one figure): oracle, ridge and Chronos-2.
- Takeaway framing: a null on about 65 days rules out *large* edges only.
  State the detection limit as a number.

## Results: the blind window, opened once

Verdict first, per the style guide.

- One sentence on whether any cell met the SMI and what SPA's p-value was.
- The 6-cell table: MAE ratio, 95% interval, whether the cell meets the SMI.
  Then the h = 20 descriptive row.
- Go through the predictions: which held and which didn't, *including any
  that were wrong.*
- Caveats: small n, the detection limit, one asset, one model, daily only.
  Note the paper's daily-frequency result (Kronos rank signal that didn't
  survive trading or a later sample) as context, not confirmation.

## What this tells us

- Cautious generalization: on near-random-walk series, the default is no
  skill. A claimed win has to beat drift, beat luck across all the
  comparisons made, and hold up on data the analyst hadn't seen.
- Turn it back on the series: Posts 01–04 reported pairwise intervals. Say
  what changes going forward (state the family, use SPA where many cells
  compete, include an ingredient control).
- Close on a short line built around "what does nothing look like?"

## Figures and tables (planned)

1. Post 04's h = 20 bar, and the same comparison against drift (the hook
   paid off).
2. Price-level scatter: "same as today" against Chronos-2 (intuition 2).
3. Pairwise wins against SPA, seen window (the multiplicity lesson).
4. Detection-limit curve (intuition 7).
5. Blind-window MAE-ratio table with intervals (the verdict).

## References

- Upadhyay, R. (2026). *Do Time-Series Foundation Models Forecast Index-Futures
  Returns? A Preregistered Evaluation of Models, Metrics, and Input
  Representations.* arXiv:2611.12141.
- Wan, S., Ma, M., et al. (2026). *Forecast Collapse in Time-Series Foundation
  Models.* arXiv:2608.14106.
- Hansen, P. R. (2005). A test for superior predictive ability. *JBES* 23(4).
- Ansari, A. F., et al. Chronos-2 (arXiv:2510.15821; also listed as 2610.41956).
- Optional: Noguer i Alonso & Franklin (2026), arXiv:2606.27100. Daily equity
  returns, Chronos-2 among the models tested, "small and sparse" gains. Cited
  by Upadhyay; **not yet verified here**, so check it before citing.
