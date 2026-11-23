# Baseline and classical models

All candidates use the frozen shared training and validation clients. No test predictions are made here.
Each candidate fits its own identical preprocessing pipeline on training data only. The serialized pipeline takes raw named predictors.

## Search and selection

The predeclared search has one majority-class dummy, three logistic C values, and three histogram gradient boosting configurations.
Logistic regression uses lbfgs, at most 2,000 iterations, no class weighting, and L2 regularization. Every run checks convergence.
Histogram gradient boosting disables internal early stopping; its 200 boosting iterations are fixed in advance. External validation AP selects the configuration.
AP ties select the first predeclared configuration. Working thresholds maximize validation F1, with the highest threshold used for ties. The dummy stays at 0.5 to preserve its majority-class behavior.

## Actual selected validation results

- **dummy**: AP 0.221284; ROC-AUC 0.500000; working threshold 0.500000; F1 0.000000. Parameters: `{"strategy": "most_frequent"}`.
- **logistic**: AP 0.548352; ROC-AUC 0.778008; working threshold 0.250129; F1 0.557988. Parameters: `{"C": 0.1}`.
- **boosting**: AP 0.555499; ROC-AUC 0.787634; working threshold 0.339053; F1 0.559705. Parameters: `{"l2_regularization": 1.0, "learning_rate": 0.1, "max_iter": 200, "max_leaf_nodes": 15}`.

The complete search and both threshold metrics are in `artifacts/classical/validation_search.csv` and `selected.json`.
The most-frequent dummy gives every client the same score, so its AP equals validation prevalence and its ROC-AUC is 0.5.
All three saved pipelines were reloaded and reproduced validation scores exactly.

## Reproduce

Run `python -m src.classical` after data download and split preparation. CPU thread pools are limited to the shared configuration (2 threads).

A model score is an estimated probability, not a guaranteed calibrated default risk. Threshold optimization is a teaching proxy and does not encode measured review capacity or real error costs.
