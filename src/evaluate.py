"""Final held-out evaluation. Never choose models or thresholds from test data."""
from __future__ import annotations

import json
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve
from threadpoolctl import threadpool_limits

from src.common import ROOT, config, evaluate_scores, write_json
from src.freeze import verify_frozen
from src.preprocessing import load_partitions

OUT = ROOT / "artifacts/evaluation"
LABELS = {"dummy": "Majority baseline", "logistic": "Logistic regression",
          "boosting": "Gradient boosting", "mlp": "PyTorch MLP"}
COLORS = {"dummy": "#808080", "logistic": "#b55a30", "boosting": "#17657c", "mlp": "#76579b"}


def compute_bootstrap_ap_ci(y_true, scores, n_bootstraps=1000, seed=42, alpha=0.05):
    """Calculate 95% bootstrap confidence interval for Average Precision."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boot_aps = []
    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(y_true[idx])) > 1:
            boot_aps.append(float(average_precision_score(y_true[idx], scores[idx])))
    lower = float(np.percentile(boot_aps, 100 * (alpha / 2)))
    upper = float(np.percentile(boot_aps, 100 * (1 - alpha / 2)))
    return lower, upper


def compute_paired_bootstrap(y_true, predictions, reference="boosting", others=("logistic", "mlp"),
                             n_bootstraps=1000, seed=42, alpha=0.05):
    """Paired bootstrap of AP differences (reference - other) on the SAME resampled clients.

    Each replicate draws one set of test indices and scores every model on it, so the
    correlation between models' errors is preserved. Used for description only.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = {other: [] for other in others}
    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        ref_ap = average_precision_score(y_true[idx], predictions[reference][idx])
        for other in others:
            diffs[other].append(float(ref_ap - average_precision_score(y_true[idx], predictions[other][idx])))
    short = {"boosting": "gb", "logistic": "lr", "mlp": "mlp"}
    return {f"diff_{short[reference]}_{short[other]}": {
                "mean": float(np.mean(values)),
                "ci_lower": float(np.percentile(values, 100 * (alpha / 2))),
                "ci_upper": float(np.percentile(values, 100 * (1 - alpha / 2)))}
            for other, values in diffs.items()}


def load_model(name, metadata):
    if name == "mlp":
        from src.neural import load_predictor
        return load_predictor()
    return joblib.load(ROOT / metadata["model_path"])


def plot_precision_recall(y, predictions):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.3, 5))
    for name, scores in predictions.items():
        if name == "dummy":
            continue  # A horizontal prevalence line is clearer than a tied-score staircase.
        precision, recall, _ = precision_recall_curve(y, scores)
        ap = average_precision_score(y, scores)
        ax.step(recall, precision, where="post", color=COLORS[name],
                label=f"{LABELS[name]} (AP {ap:.3f})", linewidth=1.7)
    ax.axhline(np.mean(y), linestyle="--", color=COLORS["dummy"],
               label=f"Prevalence / dummy AP ({np.mean(y):.3f})")
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall", ylabel="Precision",
           title="Held-out clients: precision–recall")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(OUT / "precision_recall.png", dpi=180)
    plt.close(fig)


def plot_confusions(metrics):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 2, figsize=(8, 12))
    for row, name in enumerate(LABELS):
        for col, threshold_type in enumerate(["0.5", "working"]):
            ax = axes[row, col]
            subset = metrics[(metrics.model == name) & (metrics.threshold_type == threshold_type)]
            if subset.empty:
                ax.axis("off")
                continue
            item = subset.iloc[0]
            matrix = np.array([[item.tn, item.fp], [item.fn, item.tp]], dtype=int)
            ax.imshow(matrix, cmap="Blues", vmin=0, vmax=int(matrix.max()))
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                            color="white" if matrix[i, j] > matrix.max()/2 else "black", fontsize=12)
            ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["No default", "Default"],
                   yticklabels=["No default", "Default"], xlabel="Predicted", ylabel="Actual",
                   title=f"{LABELS[name]} | {threshold_type}: {item.threshold:.3f}")
    fig.suptitle("Test confusion matrices: fixed validation decisions", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, .98])
    fig.savefig(OUT / "confusion_matrices.png", dpi=180)
    plt.close(fig)


def select_examples(x, y, scores, threshold):
    """Fixed descriptive sampling, not a new model-selection step.

    Sort each outcome category by score, then sample evenly-spaced positions.
    This shows low/middle/high score errors rather than only spectacular cases.
    """
    pred = (scores >= threshold).astype(int)
    category = np.where(y == 1, np.where(pred == 1, "TP", "FN"), np.where(pred == 1, "FP", "TN"))
    result = x.copy()
    result.insert(0, "row_index", x.index.to_numpy())
    result.insert(1, "error_type", category)
    result.insert(2, "actual", y)
    result.insert(3, "probability", scores)
    result.insert(4, "threshold", threshold)
    result.insert(5, "prediction", pred)
    chosen = []
    for kind, count in [("FP", 3), ("FN", 3), ("TP", 2), ("TN", 2)]:
        subset = result[result.error_type == kind].sort_values(["probability", "row_index"])
        if subset.empty:
            continue
        quantiles = [.1, .5, .9] if count == 3 else [.25, .75]
        positions = np.unique(np.rint(np.array(quantiles) * (len(subset)-1)).astype(int))
        chosen.append(subset.iloc[positions])
    examples = pd.concat(chosen, ignore_index=True)
    examples.to_csv(OUT / "error_examples.csv", index=False)
    return examples


def permutation_analysis(model, x, y, seed, repeats=5):
    """AP drop after shuffling each original raw column; interpretation only."""
    import matplotlib.pyplot as plt
    baseline = average_precision_score(y, model.predict_proba(x)[:, 1])
    rng = np.random.default_rng(seed)
    rows = []
    for column in x.columns:
        drops = []
        for _ in range(repeats):
            perturbed = x.copy()
            perturbed[column] = rng.permutation(perturbed[column].to_numpy())
            drops.append(float(baseline-average_precision_score(y, model.predict_proba(perturbed)[:, 1])))
        rows.append({"feature": column, "mean_ap_drop": float(np.mean(drops)),
                     "std_ap_drop": float(np.std(drops)), "repeats": repeats,
                     **{f"repeat_{i+1}": score for i, score in enumerate(drops)}})
    importance = pd.DataFrame(rows).sort_values("mean_ap_drop", ascending=False)
    importance.to_csv(OUT / "permutation_importance.csv", index=False)
    top = importance.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.3, 5))
    ax.barh(top.feature, top.mean_ap_drop, xerr=top.std_ap_drop, color="#17657c", capsize=3)
    ax.axvline(0, color="gray", linewidth=.8)
    ax.set(xlabel="Mean decrease in test AP (5 permutations)", ylabel="Original feature",
           title="Selected model: permutation importance")
    ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    fig.savefig(OUT / "permutation_importance.png", dpi=180)
    plt.close(fig)
    return importance


def write_report(frozen, metrics, examples, importance, size):
    winner = frozen["selected_model"]
    work = metrics[(metrics.model == winner) & (metrics.threshold_type == "working")].iloc[0]
    standard = metrics[(metrics.model == winner) & (metrics.threshold_type == "0.5")].iloc[0]
    model_ap = metrics.groupby("model")["ap"].first()
    neural = frozen["models"]["mlp"]
    history = pd.read_csv(ROOT / "artifacts/neural/history_selected.csv")
    best = history.loc[history.epoch == neural["best_epoch"]].iloc[0]
    last = history.iloc[-1]
    texts = [
        "# Final evaluation and review", "",
        f"The chosen model is **{LABELS[winner]}**, selected before test scoring by validation AP {frozen['models'][winner]['validation_ap']:.6f}. Settings, thresholds and checkpoint hashes were locked in artifacts/frozen_selection.json and verified before loading test partitions.",
        f"The shared test contains {size:,} held-out records. Positive prevalence is {work.positive_rate:.6f}; this is also the majority dummy's AP. All 30,000 records remain in the study, with exact predictor duplicates confined to one partition.",
        "", "## Metrics and operating point", "",
        "The definitive comparison is artifacts/evaluation/test_metrics.csv (one row per model/threshold). AP and ROC-AUC use continuous scores; threshold metrics use score >= threshold. Zero precision when no positives are predicted is explicitly defined as zero. AP is not accuracy or trapezoidal PR area.",
    ]
    for name in LABELS:
        row = metrics[(metrics.model == name) & (metrics.threshold_type == "working")].iloc[0]
        texts.append(f"- {LABELS[name]}: test AP {row.ap:.6f}; ROC-AUC {row.roc_auc:.6f}; threshold {row.threshold:.6f}; precision {row.precision:.6f}; recall {row.recall:.6f}; F1 {row.f1:.6f}. TN/FP/FN/TP = {int(row.tn)}/{int(row.fp)}/{int(row.fn)}/{int(row.tp)}.")
    texts += ["", f"For the selected model, threshold 0.5 yields precision {standard.precision:.4f}, recall {standard.recall:.4f}, F1 {standard.f1:.4f}, {int(standard.fp)} false positives and {int(standard.fn)} missed defaults. The validation-selected threshold {work.threshold:.6f} yields precision {work.precision:.4f}, recall {work.recall:.4f}, F1 {work.f1:.4f}, {int(work.fp)} false positives and {int(work.fn)} missed defaults.",
              f"The review queue changes from {int(standard.tp+standard.fp)} to {int(work.tp+work.fp)} cases out of {size}. F1 was fixed as the threshold-selection objective before test evaluation. It balances precision and recall, but does not encode real review capacity or asymmetric financial costs. A lower threshold usually catches more defaults while sending more non-default cases for review. The threshold is not reoptimized on test.",
              "", "## Case inspection", "",
              "error_examples.csv contains three false positives, three false negatives, two true positives and two true negatives when each category exists. Examples are chosen at fixed score positions within each outcome group (10/50/90% for errors, 25/75% for correct cases), not to imply that one case explains the entire population. The source row index is a reproducibility key and is never an input to the model."]
    for r in examples.itertuples():
        status = f"documented {int(r.PAY_0)}-month payment delay" if r.PAY_0 >= 1 else ("documented duly-paid status" if r.PAY_0 == -1 else f"code {int(r.PAY_0)} (semantics not specified by UCI)")
        outcome = {"FP": "flagged for review but did not default", "FN": "not prioritized but defaulted", "TP": "flagged and defaulted", "TN": "not prioritized and did not default"}[r.error_type]
        texts.append(f"- {r.error_type}, row {int(r.row_index)}: score {r.probability:.4f}, actual {int(r.actual)}; {outcome}. Latest repayment status: {status}; credit limit NT$ {int(r.LIMIT_BAL):,}, latest bill NT$ {int(r.BILL_AMT1):,}, latest payment NT$ {int(r.PAY_AMT1):,}. These observed values describe the case; they do not prove why the model assigned its score or why the client defaulted.")
    texts += ["", "False positives show that risky-looking histories do not guarantee a future default. False negatives show that recorded payment history cannot fully identify future events. Income changes and other unobserved circumstances cannot be inferred from these examples.",
              "", "## Model interpretation", "",
              "Permutation importance is calculated after model selection on the unchanged test set for description only. Each original column is shuffled five times with seed 42; the model is never refitted. The saved value is the mean decrease in AP; error bars are the standard deviation across permutations, not a confidence interval over clients.",
              "The largest decreases are: " + "; ".join(f"{r.feature} ({r.mean_ap_drop:.4f})" for r in importance.head(5).itertuples()) + ".",
              "This measures predictive reliance, not causality. Correlated monthly histories can substitute for one another, distributing or understating individual importance. Shuffling may create unusual feature combinations. Negative small values can reflect finite-sample/permutation variation. This interpretation does not trigger another tuning round.",
              "", "## Neural learning and model comparison", "",
              f"The selected MLP restores epoch {neural['best_epoch']} of {neural['epochs_run']} epochs run. At that checkpoint train AP = {best.train_ap:.6f}, validation AP = {best.validation_ap:.6f}, train BCE = {best.train_loss:.6f}, validation BCE = {best.validation_loss:.6f}. At the last epoch the corresponding values are {last.train_ap:.6f}, {last.validation_ap:.6f}, {last.train_loss:.6f}, and {last.validation_loss:.6f}.",
              "Training and validation curves are measured with dropout disabled. Improving train scores alongside stalled/worsening validation scores indicate overfitting; restoring the earlier validation-AP checkpoint limits it. Loss and AP can peak at different epochs because BCE assesses probabilistic fit while AP assesses ranking. Validation sampling variation also affects small fluctuations. A persistent low train and validation score would suggest underfitting; the saved curves should be read rather than assuming it from architecture alone.",
              f"The MLP test AP is {model_ap['mlp']:.6f}, compared with boosting {model_ap['boosting']:.6f} and logistic regression {model_ap['logistic']:.6f}. " + ("In this run the MLP trails boosting. Trees can efficiently capture threshold effects and interactions in tabular data, while this compact MLP search covered only four configurations with a single seed. This is a plausible explanation, not proof that neural networks are unsuitable." if model_ap['mlp'] < model_ap['boosting'] else "In this run the MLP matches or exceeds boosting on test AP. This does not establish general neural superiority; the model used in the demo still follows the frozen validation decision."),
              "The linear logistic baseline has a restricted additive decision function on transformed features, so it cannot freely model interactions. The majority baseline deliberately ignores all information and illustrates why accuracy alone would be misleading with class imbalance. A single split and small search do not establish a statistically significant universal ranking.",
              "", "## Limitations and realistic next steps", "",
              "The data describe historical Taiwanese credit-card clients in 2005. Evaluation is on held-out clients from this source population, not a later month, new country or new bank. The label is the workbook's next-month-default outcome; the UCI description does not provide a detailed operational delinquency definition. Features must be available by the end of September for this prediction boundary.",
              "A future-period study needs multiple timestamped cohorts, leakage-safe feature cutoffs, consistent outcome windows and a forward temporal test. A bank-transfer study needs an external bank's compatible schema and independently labeled holdout. Both require drift and calibration assessment, subgroup performance/fairness checks, and operational review-capacity/cost analysis. Undefined codes and demographic fields deserve extra care. We do not claim calibrated individual risk, causality, fairness, or readiness for automated lending decisions.",
              "", "## Verification", "",
              "Frozen fingerprints verified before test loading. Every saved model was reloaded for test inference. Scores are checked finite and within [0,1]. Confusion counts sum to test size. All models use the identical test row indices. No settings are selected in this module."]
    (ROOT / "docs/evaluation.md").write_text("\n\n".join(texts)+"\n", encoding="utf-8")


def evaluate():
    frozen = verify_frozen()  # Mandatory gate BEFORE test loading or scoring.
    import matplotlib
    matplotlib.use("Agg")
    cfg = config()
    OUT.mkdir(parents=True, exist_ok=True)
    x, y = load_partitions(include_test=True)["test"]
    metrics_rows, predictions, models = [], {}, {}
    with threadpool_limits(limits=cfg["threads"]):
        for name in LABELS:
            meta = frozen["models"][name]
            model = load_model(name, meta)
            scores = model.predict_proba(x)[:, 1]
            assert scores.shape == y.shape and np.isfinite(scores).all()
            assert ((scores >= 0) & (scores <= 1)).all()
            ci_low, ci_high = compute_bootstrap_ap_ci(y, scores, seed=cfg["seed"]) if name != "dummy" else (float(np.mean(y)), float(np.mean(y)))
            predictions[name], models[name] = scores, model
            thresholds = [("working", .5)] if name == "dummy" else [("0.5", .5), ("working", meta["threshold"])]
            for label, threshold in thresholds:
                row = {"model": name, "threshold_type": label, **evaluate_scores(y, scores, threshold),
                       "ap_ci_lower": ci_low, "ap_ci_upper": ci_high}
                assert sum(row[k] for k in ["tn", "fp", "fn", "tp"]) == len(y)
                metrics_rows.append(row)
        metrics = pd.DataFrame(metrics_rows)
        metrics.to_csv(OUT / "test_metrics.csv", index=False)
        write_json(OUT / "test_metrics.json", metrics_rows)
        pd.DataFrame({"row_index": x.index, "actual": y, **predictions}).to_csv(OUT / "test_predictions.csv", index=False)
        write_json(OUT / "paired_bootstrap.json", compute_paired_bootstrap(y, predictions, seed=cfg["seed"]))
        plot_precision_recall(y, predictions)
        plot_confusions(metrics)
        winner = frozen["selected_model"]
        examples = select_examples(x, y, predictions[winner], frozen["models"][winner]["threshold"])
        importance = permutation_analysis(models[winner], x, y, cfg["seed"])
    write_report(frozen, metrics, examples, importance, len(y))
    write_json(OUT / "summary.json", {"evaluated_utc": datetime.now(timezone.utc).isoformat(),
               "selected_model": winner, "selection_source": "frozen validation AP",
               "test_size": len(y), "prevalence": float(np.mean(y)), "frozen_utc": frozen["created_utc"],
               "threshold_rule": cfg["threshold_rule"], "no_test_tuning": True,
               "permutation_repeats": 5, "seed": cfg["seed"]})
    print(metrics.to_string(index=False), flush=True)
    return metrics


if __name__ == "__main__":
    evaluate()
