# Course requirements and evidence

Source: user-supplied *MLA Final Project Instructions*, 3 pages, extracted in `docs/course_requirements.txt`. The source PDF is an assessment document; the user's request specifies the concrete topic, models, metrics, CPU execution and deliverables. This is an evidence checklist, not a self-awarded grade.

## Problem formulation and motivation — 10%

- Binary classification: predict the supplied next-month default label from 23 end-of-observation-period features.
- Motivation: prioritize clients for additional review; output a score and a threshold-based review decision.
- Success measure: validation AP selects models; held-out AP and ROC-AUC measure ranking, precision/recall/F1 describe decisions.
- Evidence: README, presentation opening, `config.json`, `src/common.py`.

## Data understanding and exploratory analysis — 15%

- Real public 30,000-client UCI dataset, attribution/license, source hashes and schema validation.
- Feature groups, time availability and underdocumented categories explained without assigning unsupported meanings.
- Class balance, missingness, duplicate profiles and conflicting labels, unusual codes, monetary tails and payment-delay patterns examined.
- Train-only statistical EDA with implications for preprocessing and metrics. Privacy, historical/geographic sampling and use-case limitations discussed.
- Evidence: `src/data.py`, `docs/data.md`, `data/raw/provenance.json`, `artifacts/eda/`.

## Preprocessing and evaluation design — 15%

- ID excluded; all 30,000 rows retained, duplicate predictor groups cannot cross partitions.
- Frozen approximately 70/15/15 stratified group split, seed 42, full row coverage and fingerprints.
- One-hot encoding and numeric scaling fit on train only; signed log handles negative and long-tailed monetary values.
- Validation selects hyperparameters, neural checkpoint and threshold. Test is scored after frozen settings; no test-directed revisions.
- No temporal-generalization claim from a single-period client split.
- Evidence: `src/preprocessing.py`, `docs/protocol.md`, `tests/test_protocol.py`, `src/freeze.py`, `artifacts/splits/`, `artifacts/frozen_selection.json`.

## Model implementation, neural network and tuning — 30%

- Majority-class baseline; logistic regression; histogram gradient boosting; genuinely trained PyTorch MLP.
- Shared inputs and train-fitted preprocessing, CPU execution, seven classical candidates including baseline and two neural candidates.
- Two hidden layers, ReLU, dropout, one logit, BCEWithLogitsLoss, AdamW, batch size, learning rate, weight decay, epochs and patience documented.
- Saved best-validation-AP neural checkpoint, complete learning histories and candidate results. Reload checks compare saved and in-memory scores.
- Evidence: `config.json`, `src/classical.py`, `src/neural.py`, `docs/classical.md`, `docs/neural.md`, model artifacts.

## Results, interpretation and critical analysis — 20%

- Shared test comparison and PR curve; AP prevalence reference; ROC-AUC; fixed-threshold precision/recall/F1/confusion matrices.
- Standard 0.5 versus validation-selected threshold, concrete false positives/negatives and true outcomes.
- Original-feature permutation importance, predictive versus causal interpretation and correlated-feature caveat.
- Neural learning curves, overfitting/underfitting discussion, limited tuning budget, historical dataset limitations and realistic external validation next steps.
- Evidence: `src/evaluate.py`, `artifacts/evaluation/`, `docs/evaluation.md`, presentation results slides.

## Presentation and defense — 10%

- Editable English PPTX with actual artifact-derived metrics; concise workflow and demo.
- Speaking script and 15 questions/answers cover AP, precision/recall, leakage, backpropagation, model choice, failed candidates and generalization.
- Exact demonstration command reloads selected model; actual target appears after prediction.
- Sources, AI contribution disclosure, student-name and contribution placeholders. Maximum 3 students; every member should explain the overall workflow. No phones in hand during defense.
- Evidence: `presentation/`, `docs/speaker_notes.md`, `docs/defense.md`, `docs/sources.md`, `src/demo.py`.

## Reproducibility and verification

- `README.md`, pinned dependencies, seed/configuration, cached original source, structured scripts, CSV/JSON metrics and saved models are supplied.
- Execution and presentation checks are recorded in `artifacts/verification.json` after the final run.
- A presentation preview is supplementary; the editable PPTX remains the submission deck.
- Remaining human action: fill in participant names and truthful human contributions; rehearse and understand the defense. No assessment score is guaranteed.
