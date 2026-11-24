# Frozen experimental protocol

This protocol is established before model selection and final test evaluation.
The only random seed is **42**. `config.json` sets `device: cuda`; the MLP falls back to CPU when CUDA is unavailable. Classical models always run on CPU.

## Prediction time and information boundary

Each row describes a client using account information and six months of payment
history (April–September 2005). The target is the source file's next-month
default-payment indicator (1 = default, 0 = no default). Predict at the end of
September using these past/present fields; the next-month outcome must never be
an input. UCI does not specify a precise operational days-past-due definition;
do not claim that the label means 90-day default. ID is metadata, never a feature.

## One shared split

`src/preprocessing.py:create_splits` creates **approximately 70/15/15**
train/validation/test partitions using 20 shuffled stratified group folds, seed
42, with folds 0–13 used for train, 14–16 for validation and 17–19 for test.
No seed search is performed. Sizes and prevalence are recorded in
`artifacts/splits/manifest.json`; integer row indices are saved as NPZ and JSON.
A SHA-256 fingerprint binds those indices to the exact canonical CSV.

Groups are defined by exact equality of the **23 predictor values**. Both exact
duplicate records and feature-identical records with different labels must stay
together. Outcome is deliberately excluded from the grouping key. All 30,000
observations are retained: repeated profiles need not be clerical errors, and
conflicting outcomes are informative about prediction uncertainty. Keeping
profiles together avoids an overly easy cross-split memorization opportunity.
This grouping requirement can make the sizes and class proportions differ
slightly from exact 70/15/15 stratification.

The source-wide schema/duplicate audit and label-stratified assignment are data
construction steps. Outcome-dependent EDA is restricted to train. Test labels
are not used to select transformations, candidate settings, checkpoints,
thresholds, or the final model. The training API excludes test unless
`load_partitions(include_test=True)` is explicitly requested for final reporting.

This estimates performance on held-out clients/profiles from this historical
dataset. It is **not** temporal validation or external-bank validation. Those
would require newer dated cohorts or a separate bank, an unchanged feature and
label definition, chronological/entity-aware splitting, and checks for shifted
prevalence, calibration, performance, and subgroup behavior.

## Train-only preprocessing

All models receive the same raw split rows and use the shared preprocessing
factory. Fit the preprocessor on `X_train`; call `transform` on validation and
test. Predictors are explicitly selected, so ID and target cannot enter.

- SEX, EDUCATION, MARRIAGE and six PAY status fields are one-hot encoded.
  PAY -2 and 0, EDUCATION 0/5/6 and MARRIAGE 0 are retained as distinct observed
  codes, without inventing undocumented semantic meanings. This avoids assuming
  a meaningful numeric distance between nonstandard status categories.
- Encoder category vocabulary is learned only from train. An unseen category
  produces all zeros for that field (`handle_unknown='ignore'`).
- All monetary values use a fixed signed log transform,
  `sign(x) * log1p(abs(x))`, to reduce heavy tails while retaining negative bills.
  AGE stays in years before scaling. Means and variances for all numeric
  columns are fitted on train with StandardScaler. The output is dense float32.
- No missing-value imputer is fitted: the original dataset has no missing
  values and the loader fails if unexpected missingness is encountered.

## Selection, threshold, and final test

**Average Precision (AP)** on validation probability scores selects candidate
configurations and the final model. ROC-AUC is secondary. Checkpoints and early
stopping for the MLP use validation only. All models use identical held-out rows.

Before any test evaluation, freeze each learned model's operating threshold:
choose the distinct validation score that maximizes validation F1, breaking an
exact tie with the **highest threshold**. A predicted score equal to the
threshold is positive (`p >= threshold`). This is an explicit educational
compromise without a known business cost matrix. It does not optimize a real
bank's cost or review capacity. Compare this threshold to 0.5 on test, without
adjusting it after seeing test performance. The majority DummyClassifier stays
at 0.5 as a clear always-negative baseline.

Save settings, scores and selected thresholds before final evaluation. Then
report AP, ROC-AUC, precision, recall, F1, TN/FP/FN/TP, and the positive-class
prevalence as the no-skill AP reference. Review errors and permutation importance
after decisions are frozen; do not use these analyses to retune models.

## Verification and defense entry points

Run `python -m src.preprocessing check` and
`python -m unittest discover -s tests`. Tests check complete row coverage,
zero row/group overlap, train-only feature exposure and scaling, and stable
encoder shape for unknown validation categories.

During the defense open `create_splits`, `load_partitions`, and
`make_preprocessor` in `src/preprocessing.py`.
