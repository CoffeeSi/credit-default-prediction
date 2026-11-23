# Predicting Next-Month Credit Card Default Using Payment History

An executed, GPU-accelerated (NVIDIA RTX 4050 / CUDA) educational binary classification project using 30,000 real UCI credit-card clients. It prioritizes cases for additional review; it is not an automated lending decision system.

## Setup

The reference environment is Windows, Python 3.12, GPU (NVIDIA RTX 4050 / CUDA), seed 42. Exact installed versions are pinned in `pyproject.toml`, `uv.lock`, and `requirements.txt`.

### Быстрый запуск через `uv` (рекомендуется):

```powershell
cd D:\AITU\ml\final
uv run python run.py
```

Для повторной проверки с готовыми обученными моделями:
```powershell
uv run python run.py --skip-training
```

### Запуск через `make` (Makefile):

```powershell
make help           # Список всех доступных команд
make all            # Полный цикл обучения и оценки
make pipeline-fast  # Повторная проверка с готовыми весами
make test           # Запуск юнит-тестов
make demo           # Живой запуск инференса на клиенте #0
make presentation   # Сборка презентации PowerPoint (.pptx)
make verify         # Полная валидация целостности проекта
```

### Запуск через стандартный Python (.venv):

```powershell
cd D:\AITU\ml\final
.\.venv\Scripts\python.exe run.py
```

The final command downloads/caches UCI data, constructs the fixed split, produces training-only EDA, fits every model, freezes validation decisions, verifies the protocol, evaluates test data, demonstrates inference, and builds the presentation. Internet is needed only for package installation and the first data download. The cached raw XLS/ZIP and trained artifacts are included. All paths in Python resolve from the project location; the folder may be moved.

To reuse supplied models and rebuild final outputs:

```powershell
.\.venv\Scripts\python.exe run.py --skip-training
```

## Individual stages

```powershell
.\.venv\Scripts\python.exe -m src.data download
.\.venv\Scripts\python.exe -m src.preprocessing prepare
.\.venv\Scripts\python.exe -m src.data eda
.\.venv\Scripts\python.exe -m src.classical
.\.venv\Scripts\python.exe -m src.neural train
.\.venv\Scripts\python.exe -m src.freeze
.\.venv\Scripts\python.exe -m unittest discover -s tests
.\.venv\Scripts\python.exe -m src.evaluate
.\.venv\Scripts\python.exe -m src.demo --index 0 --reveal-label
.\.venv\Scripts\python.exe scripts/build_presentation.py
```

On macOS/Linux replace `.\.venv\Scripts\python.exe` with `.venv/bin/python`. The validated environment is Windows; other operating systems may differ slightly numerically.

## Experiment rules fixed before test evaluation

- Target: the supplied next-month default label, 1 = default and 0 = no default. Predictors describe April–September 2005; the next month is October by the workbook label and observation dates. UCI does not specify a precise operational days-past-due rule for the outcome.
- Source: [UCI dataset 350](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients), Yeh (2009), [DOI](https://doi.org/10.24432/C55S3H), CC BY 4.0. Preserve attribution. See `docs/data.md` and raw-data provenance for codes, license and checksums.
- `ID` is never a predictor. All 23 predictors are used identically across learned models. The original row order and all 30,000 records are retained.
- Seed 42; 20 stratified group folds assigned 14/3/3 for approximately 70/15/15. Exact duplicate predictor profiles stay together, including contradictory target labels. Saved indices and fingerprints prevent accidental changes.
- Statistical EDA uses train. Only schema/duplicate integrity and stratification use the full dataset. One-hot encoders and numeric scaling fit only on train. Monetary variables use a signed logarithm; rare/undefined category codes remain distinct without invented semantics. Unknown inference categories map to an all-zero encoding for that feature.
- Compare majority-class DummyClassifier, logistic regression (3 C values), histogram gradient boosting (3 configurations) and PyTorch MLP (4 configurations on GPU). `config.json` is the authoritative search specification.
- Select within each family and choose the deployment/demo model by highest validation AP. For learned models, select the validation threshold maximizing F1, breaking ties in favor of the highest threshold. Dummy retains 0.5. This is an educational operating rule, not an estimated economic optimum.
- Save `artifacts/frozen_selection.json` with settings and SHA-256 hashes before scoring test. Final evaluation verifies those hashes. No retraining on train+validation takes place.
- Report AP and ROC-AUC from continuous scores; precision, recall, F1 and confusion counts at 0.5 and validation-selected thresholds. AP's no-skill reference is positive prevalence. All reported test results share the same clients.

## Quick code walkthrough

1. `src/data.py`: official data download, schema checks and EDA.
2. `src/preprocessing.py`: `create_splits`, `check_splits`, `make_preprocessor`, `load_partitions`.
3. `src/classical.py`: baseline and classical candidate search.
4. `src/neural.py`: MLP architecture, forward pass, loss, `backward`, optimizer update, validation checkpoint and early stopping.
5. `src/common.py`: `evaluate_scores` and `choose_threshold`.
6. `src/freeze.py`: decision lock and artifact verification.
7. `src/evaluate.py`: final comparison, PR curve, errors and interpretation.
8. `src/demo.py`: reload selected model and predict a held-out client.

## Demonstration

```powershell
uv run python -m src.demo --index 0 --reveal-label
# или
.\.venv\Scripts\python.exe -m src.demo --index 0 --reveal-label
```

The index is a position in the frozen test partition. The command prints estimated default probability, validation-selected threshold, and review decision. It reveals the actual target separately afterward. Omit `--reveal-label` to keep the label hidden. The predictor accepts a raw dataframe with the 23 named features; it loads the saved train-fitted preprocessing automatically.

## Submission materials

- `presentation/`: editable English PowerPoint and rendered preview when available.
- `docs/speaker_notes.md`, `docs/defense.md`: speaking script and defense Q&A.
- `docs/rubric_checklist.md`: course PDF requirements mapped to evidence.
- `docs/data.md`, `docs/protocol.md`, `docs/classical.md`, `docs/neural.md`, `docs/evaluation.md`: reasoning and actual results.
- `docs/sources.md`: references, academic AI disclosure, and verified student team members/contributions (Yevgeniy Averyanov, Madiyar Amanzhol, Aldiyar Amangeldi).
- `artifacts/`: fixed split, search results, saved weights/pipelines, frozen selection, metrics, plots and verification logs.

## Interpretation and limitations

This split estimates performance on held-out clients from the supplied historical dataset. It is not a temporal backtest and does not establish transfer to another bank. A later-period or external-bank test needs comparable timestamped features, matched outcome definitions and observation windows, client/group separation, and assessment of drift, calibration and subgroup errors. Probability scores have not been independently calibrated. F1 does not encode a review team's capacity or unequal real error costs. Demographic variables raise fairness considerations that this experiment does not resolve. Correlated payment-history columns can share permutation importance; importance is predictive, not causal. These are empirical results from one split and a small tuning budget, not a universal ranking of algorithms.

Saved joblib and PyTorch files should only be loaded from trusted sources. Students should rehearse the demo, explain the code and fill in the contribution fields honestly. The PDF also prohibits holding phones during defense.
