"""Train classical candidates using train/validation only; freeze selections."""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from src.common import ROOT, choose_threshold, config, evaluate_scores, write_json
from src.preprocessing import load_partitions, make_preprocessor, source_sha256

OUTPUT = ROOT / "artifacts" / "classical"


def train() -> dict:
    """Fit all candidates on train and choose each family's highest validation AP.

    Neither fitting nor candidate/threshold selection requests the test partition.
    Saved sklearn pipelines accept the original named feature dataframe.
    """
    cfg = config()
    partitions = load_partitions()
    X_train, y_train = partitions["train"]
    X_val, y_val = partitions["validation"]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    candidates = [("dummy", {"strategy": "most_frequent"})]
    candidates += [("logistic", {"C": c}) for c in cfg["logistic_C"]]
    candidates += [("boosting", p) for p in cfg["boosting_candidates"]]
    selections, best_models, rows = {}, {}, []
    validation_scores = pd.DataFrame({"row_index": X_val.index, "target": y_val})

    with threadpool_limits(limits=cfg["threads"]):
        for candidate_id, (name, params) in enumerate(candidates):
            if name == "dummy":
                estimator = DummyClassifier(**params, random_state=cfg["seed"])
            elif name == "logistic":
                estimator = LogisticRegression(**params, solver="lbfgs", max_iter=2000,
                                               random_state=cfg["seed"])
            else:
                estimator = HistGradientBoostingClassifier(**params, early_stopping=False,
                                                          random_state=cfg["seed"])
            pipeline = Pipeline([("preprocessor", make_preprocessor()), ("model", estimator)])
            start = time.perf_counter()
            pipeline.fit(X_train, y_train)
            scores = pipeline.predict_proba(X_val)[:, 1]
            seconds = time.perf_counter() - start
            threshold = 0.5 if name == "dummy" else choose_threshold(y_val, scores)
            metrics = evaluate_scores(y_val, scores, threshold)
            row = {"candidate_id": candidate_id, "model": name,
                   "params": json.dumps(params, sort_keys=True), "fit_predict_seconds": seconds,
                   **metrics}
            if name == "logistic":
                row["n_iter"] = int(estimator.n_iter_.max())
                if row["n_iter"] >= estimator.max_iter:
                    raise RuntimeError("Logistic candidate reached max_iter; resolve convergence before selection.")
            rows.append(row)
            print(f"{name} {params}: validation AP={metrics['ap']:.6f}, threshold={threshold:.6f}", flush=True)
            # Strict comparison makes a tied AP choose the first predeclared candidate.
            if name not in selections or metrics["ap"] > selections[name]["validation_ap"]:
                selections[name] = {
                    "validation_ap": metrics["ap"], "threshold": threshold,
                    "params": params, "candidate_id": candidate_id,
                    "model_path": f"artifacts/classical/{name}.joblib",
                    "validation_metrics_working": metrics,
                    "validation_metrics_0_5": evaluate_scores(y_val, scores, 0.5),
                    "seed": cfg["seed"], "source_sha256": source_sha256(),
                }
                best_models[name] = pipeline
                validation_scores[name] = scores

        for name, model in best_models.items():
            joblib.dump(model, OUTPUT / f"{name}.joblib")
            restored = joblib.load(OUTPUT / f"{name}.joblib")
            np.testing.assert_allclose(restored.predict_proba(X_val)[:, 1], validation_scores[name], rtol=0, atol=0)

    pd.DataFrame(rows).to_csv(OUTPUT / "validation_search.csv", index=False)
    validation_scores.to_csv(OUTPUT / "validation_scores.csv", index=False)
    write_json(OUTPUT / "selected.json", selections)
    _write_report(selections, rows)
    return selections


def _write_report(selections: dict, rows: list[dict]) -> None:
    lines = [
        "# Baseline and classical models", "",
        "All candidates use the frozen shared training and validation clients. No test predictions are made here.",
        "Each candidate fits its own identical preprocessing pipeline on training data only. The serialized pipeline takes raw named predictors.",
        "", "## Search and selection", "",
        "The predeclared search has one majority-class dummy, three logistic C values, and three histogram gradient boosting configurations.",
        "Logistic regression uses lbfgs, at most 2,000 iterations, no class weighting, and L2 regularization. Every run checks convergence.",
        "Histogram gradient boosting disables internal early stopping; its 200 boosting iterations are fixed in advance. External validation AP selects the configuration.",
        "AP ties select the first predeclared configuration. Working thresholds maximize validation F1, with the highest threshold used for ties. The dummy stays at 0.5 to preserve its majority-class behavior.",
        "", "## Actual selected validation results", "",
    ]
    for name, selection in selections.items():
        metric = selection["validation_metrics_working"]
        lines += [f"- **{name}**: AP {metric['ap']:.6f}; ROC-AUC {metric['roc_auc']:.6f}; working threshold {metric['threshold']:.6f}; F1 {metric['f1']:.6f}. Parameters: `{json.dumps(selection['params'], sort_keys=True)}`."]
    lines += ["", "The complete search and both threshold metrics are in `artifacts/classical/validation_search.csv` and `selected.json`.",
              "The most-frequent dummy gives every client the same score, so its AP equals validation prevalence and its ROC-AUC is 0.5.",
              "All three saved pipelines were reloaded and reproduced validation scores exactly.",
              "", "## Reproduce", "", "Run `python -m src.classical` after data download and split preparation. CPU thread pools are limited to the shared configuration (2 threads).",
              "", "A model score is an estimated probability, not a guaranteed calibrated default risk. Threshold optimization is a teaching proxy and does not encode measured review capacity or real error costs."]
    path = ROOT / "docs" / "classical.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    train()
