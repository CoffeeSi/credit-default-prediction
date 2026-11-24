# Data: origin, schema and exploratory findings

## Origin and licence

- **Dataset:** *Default of Credit Card Clients*, UCI Machine Learning Repository, dataset 350 (Yeh, 2009). DOI [10.24432/C55S3H](https://doi.org/10.24432/C55S3H).
- **Licence:** CC BY 4.0. Attribution is kept in this repository and in the presentation.
- **Population:** credit-card clients of a Taiwanese bank, observed April–September 2005.
- **Access:** `python -m src.data download` fetches the official ZIP. The ZIP, XLS and derived CSV are cached in `data/raw/` and `data/credit_default.csv`. SHA-256 checksums for all three are in `data/raw/provenance.json`.
- **Changes from the source:** the XLS header row is read, the target `default payment next month` is renamed to `default`, and the file is saved as an integer CSV. Rows, order and codes are unchanged.

## Size and schema

30,000 rows: one `ID` column (metadata, never used as a feature), 23 predictors and 1 binary target. There are no missing cells. `src/data.py::validate_schema` rejects any file that differs from this schema.

| Group | Columns | Meaning (per UCI) |
|---|---|---|
| Credit limit | `LIMIT_BAL` | Credit amount in NT$, covering the individual and family/supplementary credit |
| Demographics | `SEX`, `EDUCATION`, `MARRIAGE`, `AGE` | SEX 1 = male, 2 = female; EDUCATION 1 = graduate school, 2 = university, 3 = high school, 4 = others; MARRIAGE 1 = married, 2 = single, 3 = others; AGE in years |
| Repayment status | `PAY_0` (Sep), `PAY_2` (Aug) … `PAY_6` (Apr) | −1 = paid duly; 1…8 = payment delayed by that many months |
| Bill amount | `BILL_AMT1` (Sep) … `BILL_AMT6` (Apr) | Billed statement amount, NT$ |
| Payment amount | `PAY_AMT1` (Sep) … `PAY_AMT6` (Apr) | Amount paid, NT$ |
| Target | `default` | 1 = default next month (October 2005), 0 = no default |

**Undocumented codes.** The UCI description does not define some codes that appear in the data: `PAY_*` = −2 and 0, `EDUCATION` = 0, 5, 6, and `MARRIAGE` = 0. They are common (for example, 14,737 clients have `PAY_0 = 0`). We do not merge or relabel them. One-hot encoding keeps each code as its own category, so we never assign a meaning the source does not give. Full code counts are in `artifacts/eda/structural_audit.json`.

## Full-data structural audit (no outcome statistics)

- 0 missing cells; 30,000 unique IDs.
- 56 extra rows duplicate another row's 23 predictors. They form 52 duplicate groups covering 108 rows.
- 21 of those groups (46 rows) have **conflicting targets**: the same profile appears with different labels.
- Decision: keep every row, and keep each duplicate group inside a single partition (`StratifiedGroupKFold`). This stops the test set from containing an exact copy of a training profile.

## Training-only EDA (20,999 clients)

All outcome-related statistics and plots use the training partition only (`src/data.py::run_eda`, output in `artifacts/eda/`).

| Finding | Evidence | Modeling decision |
|---|---|---|
| **Class imbalance:** 4,645 defaults (22.1%) | `class_balance.png` | Accuracy is misleading (78% for "never default"). Average Precision is the primary metric; its no-skill level equals prevalence. Thresholds are tuned on validation rather than fixed at 0.5. |
| **Recent repayment status dominates:** the default rate is 12.8% for `PAY_0 = 0`, 17.2% for −1, 33.9% for a 1-month delay and 69.0% for a 2-month delay | `payment_status.png`, `train_pay0_default_rates.csv` | Treat `PAY_*` as categorical (one-hot). The relationship is non-linear and the undocumented codes behave differently from "paid duly". Tree models can split on these codes directly. |
| **Heavy right tails** in money columns: the median `BILL_AMT1` is NT$21,824 but the mean is NT$50,699 | `amount_distributions.png`, `train_descriptive_statistics.csv` | Use a signed log `sign(x)·ln(1+|x|)`, then a StandardScaler fitted on train only. |
| **Negative bills** (409–497 per month, probably overpayment credits) and **zero payments** (17–24% per month) | `train_summary.json` | A plain log would fail on these values; the signed log keeps both sign and zero. The values are kept, not treated as outliers. |
| **Strong correlation across months** of `PAY_*` and `BILL_AMT*` | `correlations.png` (Spearman) | Expect correlated features to share permutation importance. Interpret importance with this in mind. |

## Limitations of the data

- **Sampling and time:** clients of one bank, in one country, in 2005, shortly before Taiwan's 2005–2006 card-debt crisis. Results may not transfer to other periods, banks or markets.
- **Target definition:** UCI does not give an operational days-past-due rule for "default", so we do not claim a specific regulatory definition.
- **Undocumented codes** limit how far feature effects can be interpreted.
- **Privacy and fairness:** the data are anonymised (no names or account numbers), but they include sex, education, marital status and age. We use them as predictors only to study the task. A real lending system would need a legal review and subgroup error checks, which this project does not do.
- **Use case:** the model ranks clients for additional human review. It is not meant for automated credit decisions.
