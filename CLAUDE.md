# CLAUDE.md

Guidance for working in this repo — read this before adding a post, a demo,
or a scheduled-crawl change.

## What this repo is for

This is the working base for Eric's ongoing Medium series on emerging
time-series forecasting methods. Each post gets small, conceptual
demonstrations — not reproductions of a paper's full architecture — built
against a shared, honest rolling-origin evaluation design. The through-line
across posts is skepticism of hype paired with genuine curiosity: test
whether a method earns its place rather than assuming it does because it's
new, and report it plainly either way.

The repo also runs the topic-selection pipeline for the series: candidate
papers land in `literature/`, a weekly scheduled crawl (and manual additions)
feed `topics/backlog.md`, and a backlog entry becomes the next post's
`notebooks/<NN>-<slug>/`, and eventually a prose draft in
`posts/<NN>-<slug>.md` once the notebook's results are in — written in the
voice described below, not published from the repo (Eric copies it out to
Medium). See [topics/README.md](topics/README.md) for the full mechanics.

## Repo conventions

- **One evaluation harness, reused everywhere.** `src/evaluation.py`'s
  rolling-origin design (identical origins/horizons across models, MAE/RMSE/
  MASE/directional accuracy, a naive or seasonal-naive baseline always
  present, runtime recorded alongside accuracy) is not specific to Post 01 —
  use it for new posts rather than rebuilding evaluation logic per notebook.
- **Validation-only tuning.** Any weight, scale, or hyperparameter not fixed
  in advance gets selected on a validation slice, checked against test only
  once. Don't let a notebook pick something on test data, even informally.
- **`src/` is the shared toolbox, `notebooks/` is per-post.** Generic,
  reusable pieces (loaders, metrics, model wrappers, feature builders) go in
  `src/`. Post-specific orchestration — which models, which splits, which
  plots — stays in that post's notebook.
- **One numbered subfolder per post**, under both `notebooks/` and
  `results/`: `notebooks/<NN>-<slug>/`, `results/<NN>-<slug>/{figures,metrics}/`.
  `NN` is zero-padded and increments across the whole series, not per topic.
- **Scope is forecasting.** `posts/background/` holds assets for earlier,
  pre-series forecasting posts and isn't numbered. Posts on other topics
  (GAUSS libraries, econometrics, data cleaning) belong in the separate
  `ec78/medium-posts` repo, not here.
- **Every notebook gets a `SMALL_SCALE` toggle** near the top, cutting
  history/origins/step so it runs in a few minutes for development and CI,
  with a comment on what to flip for the full-scale numbers actually used in
  the post. Smoke-test at `SMALL_SCALE = True` before considering a notebook
  done; don't assume `False` works without running it at least once.
- **Report what actually happened, including when it's mixed or null.**
  If a validation-selected choice doesn't generalize to test, or a fancier
  model loses to the baseline, that's a finding to write up, not a bug to
  quietly fix by retuning until the numbers look better. This matters
  doubly here because it's also the blog's editorial stance (see below) —
  a notebook that only ever reports wins isn't actually usable for a post in
  this series.
- **Tests cover the parts that are easy to get subtly wrong** — causal
  no-lookahead guarantees, metric scaling, combination logic — not blanket
  coverage of every function. See `tests/` for the existing pattern.

## Writing style for blog posts

Derived from the existing posts on
[medium.com/@eclow](https://medium.com/@eclow) — match this when drafting,
rather than defaulting to a generic technical-blog voice.

**Voice.** First person, direct address to the reader ("you"). Self-position
as a practitioner or learner working through the problem, not an authority
handing down conclusions — explaining a concept the way a learner would
discover it is more in-voice than a textbook definition. Candid about
negative and messy results; frame them as informative rather than as
failures to apologize for.

**Epistemic caution.** Hedge claims proportionate to the evidence: "suggests,"
"appears," "points toward," "a defensible choice," "treat that as a
hypothesis to test." Avoid universal claims — tie conclusions to the
specific data's structure, not to a model's reputation. Call out small-sample
or statistically-insignificant results as such rather than letting a
compelling-looking number stand unqualified.

**Structure.** Hook with the question up front. Walk through method, then
results, then takeaway — don't bury the lede inside a wall of setup.
Verdict-first results sections work well: state who won or lagged, then give
precise figures with relative context (raw error paired with a % difference
or MASE-style baseline), then immediate caveats (negligible gaps, sample-size
limits), then a short "what this tells us" paragraph that generalizes
cautiously. Bold subheads, short paragraphs, and bullet lists keep technical
posts scannable.

**Sentence-level devices.** Contrastive framing ("not X, but Y"; "the
takeaway isn't... it's..."). Rhetorical questions as section transitions,
which the following text then answers. Italics for emphasis and for framing
questions. Short, punchy closing lines to end a section. Parallel-form lists
and triads. A recurring motif phrase (e.g. "earn its place") tying sections
together across a post is a good unifying device, not a cliché to avoid.
Everyday analogies help explain mechanisms (reading one letter at a time,
reading a thermometer against a baseline) — prefer these over formal
definitions when introducing a new concept.

**What to avoid.** Press-release framing, overselling a result because the
method is new or fashionable, conclusions that outrun the sample size, and
treating a single illustrative example as a substitute for the aggregate
metric table.

## Topic pipeline

See [topics/README.md](topics/README.md) for the full workflow. Short
version: a scheduled cloud agent crawls weekly for forecasting-research
developments, writes a dated digest to `topics/digests/`, adds candidates to
`topics/backlog.md` and `literature/index.md`, and opens a PR. Pick the
week's post from the backlog; manual additions to either file are always
welcome between crawls.
