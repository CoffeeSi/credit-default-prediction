"""CPU MLP: train/validation selection only, with reloadable PyTorch weights.

Defense entry point: ``train_candidate`` contains forward, backward, and update.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.common import ROOT, choose_threshold, config, evaluate_scores, write_json
from src.preprocessing import load_partitions, make_preprocessor, source_sha256

OUT = ROOT / "artifacts/neural"


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float):
        super().__init__()
        h1, h2 = hidden_dims
        self.layers = nn.Sequential(
            nn.Linear(input_dim, h1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h2, 1),
        )

    def forward(self, x):
        return self.layers(x).squeeze(-1)


def seed_everything(seed: int, threads: int, device: str = "cpu"):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if "cuda" in device and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    else:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(True)


@torch.inference_mode()
def assess(model, x, y, loss_fn, device=None):
    """Measure both datasets in eval mode, so dropout does not bias the gap."""
    model.eval()
    if device is not None:
        xb = x.to(device)
        yb = y.to(device)
    else:
        xb = x
        yb = y
    logits = model(xb)
    return float(loss_fn(logits, yb).item()), torch.sigmoid(logits).cpu().numpy()


def train_candidate(candidate_id, params, x_train, y_train, x_val, y_val, cfg):
    device = torch.device(cfg["device"] if torch.cuda.is_available() and cfg["device"] == "cuda" else "cpu")
    seed_everything(cfg["seed"], cfg["threads"], str(device))
    model = MLP(x_train.shape[1], params["hidden_dims"], params["dropout"]).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["learning_rate"],
                                 weight_decay=params["weight_decay"])
    generator = torch.Generator().manual_seed(cfg["seed"])
    loader = DataLoader(TensorDataset(x_train, y_train),
                        batch_size=cfg["mlp_batch_size"], shuffle=True,
                        generator=generator, num_workers=0)
    best_ap, best_epoch, stale, history, best_state = -1.0, 0, 0, [], None
    started = time.perf_counter()
    for epoch in range(1, cfg["mlp_max_epochs"] + 1):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)                 # forward pass
            loss = loss_fn(logits, yb)
            loss.backward()                    # chain rule -> parameter gradients
            optimizer.step()                   # AdamW uses gradients to update weights
        train_loss, train_scores = assess(model, x_train, y_train, loss_fn, device)
        val_loss, val_scores = assess(model, x_val, y_val, loss_fn, device)
        train_ap = float(average_precision_score(y_train.numpy(), train_scores))
        val_ap = float(average_precision_score(y_val.numpy(), val_scores))
        history.append({"candidate": candidate_id, "epoch": epoch,
                        "train_loss": train_loss, "validation_loss": val_loss,
                        "train_ap": train_ap, "validation_ap": val_ap})
        if val_ap > best_ap:
            best_ap, best_epoch, stale = val_ap, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        print(f"MLP {candidate_id} epoch={epoch:02d} train_AP={train_ap:.5f} val_AP={val_ap:.5f} best={best_epoch}", flush=True)
        if stale >= cfg["mlp_patience"]:
            break
    model.load_state_dict(best_state)
    _, scores = assess(model, x_val, y_val, loss_fn, device)
    checkpoint_path = OUT / f"candidate_{candidate_id}.pt"
    cpu_state = {k: v.cpu() for k, v in best_state.items()}
    torch.save({"state_dict": cpu_state, "input_dim": x_train.shape[1],
                "params": params, "best_epoch": best_epoch, "seed": cfg["seed"],
                "device": str(device)}, checkpoint_path)
    pd.DataFrame(history).to_csv(OUT / f"history_{candidate_id}.csv", index=False)
    threshold = choose_threshold(y_val.numpy(), scores)
    result = {"candidate": candidate_id, "validation_ap": best_ap,
              "threshold": threshold, "params": params, "best_epoch": best_epoch,
              "epochs_run": len(history), "model_path": checkpoint_path.relative_to(ROOT).as_posix(),
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "duration_seconds": time.perf_counter() - started,
              "metrics_0.5": evaluate_scores(y_val.numpy(), scores, 0.5),
              "metrics_working": evaluate_scores(y_val.numpy(), scores, threshold)}
    return result, history, scores


def make_curves(history, best_epoch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    frame = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric, ylabel in [(axes[0], "loss", "Binary cross-entropy"), (axes[1], "ap", "Average precision")]:
        ax.plot(frame.epoch, frame[f"train_{metric}"], label="Train", color="#124a6f")
        ax.plot(frame.epoch, frame[f"validation_{metric}"], label="Validation", color="#d17624")
        ax.axvline(best_epoch, linestyle="--", color="gray", label=f"Best AP epoch {best_epoch}")
        ax.set(xlabel="Epoch", ylabel=ylabel, title=f"MLP {ylabel.lower()}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.suptitle("Selected MLP: post-epoch evaluation with dropout disabled")
    fig.tight_layout()
    fig.savefig(OUT / "learning_curves.png", dpi=180)
    plt.close(fig)


def train():
    cfg = config()
    device_name = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device_str = "cuda" if (device_name == "cuda" and torch.cuda.is_available()) else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    partitions = load_partitions()  # no test arrays are returned
    x_train_raw, y_train = partitions["train"]
    x_val_raw, y_val = partitions["validation"]
    preprocessor = make_preprocessor()
    x_train = torch.from_numpy(preprocessor.fit_transform(x_train_raw))
    x_val = torch.from_numpy(preprocessor.transform(x_val_raw))
    joblib.dump(preprocessor, OUT / "preprocessor.joblib")
    yt, yv = torch.tensor(y_train, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)
    results, histories, scores = [], [], []
    for candidate_id, params in enumerate(cfg["mlp_candidates"]):
        result, history, predictions = train_candidate(candidate_id, params, x_train, yt, x_val, yv, cfg)
        results.append(result)
        histories.append(history)
        scores.append(predictions)
    best = max(range(len(results)), key=lambda i: results[i]["validation_ap"])
    selected = dict(results[best])
    gpu_name = torch.cuda.get_device_name(0) if device_str == "cuda" else "None"
    selected.update({"preprocessor_path": "artifacts/neural/preprocessor.joblib",
                     "input_dim": x_train.shape[1], "batch_size": cfg["mlp_batch_size"],
                     "max_epochs": cfg["mlp_max_epochs"], "patience": cfg["mlp_patience"],
                     "seed": cfg["seed"], "device": device_str, "gpu_device_name": gpu_name,
                     "threads": cfg["threads"],
                     "loss": "BCEWithLogitsLoss (unweighted)", "optimizer": "AdamW",
                     "selection": "Maximum validation AP; first candidate on exact tie",
                     "threshold_rule": cfg["threshold_rule"], "source_sha256": source_sha256(),
                     "config_sha256": hashlib.sha256((ROOT / "config.json").read_bytes()).hexdigest()})
    write_json(OUT / "search.json", results)
    write_json(OUT / "selected.json", selected)
    pd.DataFrame([{**{k: v for k, v in r.items() if k not in ["params", "metrics_0.5", "metrics_working"]},
                   **r["params"]} for r in results]).to_csv(OUT / "validation_search.csv", index=False)
    pd.DataFrame([row for hist in histories for row in hist]).to_csv(OUT / "history_all.csv", index=False)
    pd.DataFrame(histories[best]).to_csv(OUT / "history_selected.csv", index=False)
    pd.DataFrame({"row_index": x_val_raw.index, "actual": y_val, "probability": scores[best]}).to_csv(OUT / "validation_predictions.csv", index=False)
    make_curves(histories[best], selected["best_epoch"])
    predictor = load_predictor()
    reloaded = predictor.predict_proba(x_val_raw)[:, 1]
    np.testing.assert_allclose(reloaded, scores[best], rtol=0, atol=1e-5)
    write_notes(selected, histories[best])
    print(json.dumps(selected, indent=2))
    return selected


class NeuralPredictor(ClassifierMixin, BaseEstimator):
    """Raw-data sklearn-style inference wrapper for saved torch weights."""
    def __init__(self, model_path=None, preprocessor_path=None):
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        selected = json.loads((OUT / "selected.json").read_text())
        device_str = selected.get("device", "cpu")
        self.device_ = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
        if self.device_.type == "cpu":
            torch.set_num_threads(selected.get("threads", 2))
        model_file = ROOT / (model_path or selected["model_path"])
        checkpoint = torch.load(model_file, map_location=self.device_, weights_only=True)
        self.model_ = MLP(checkpoint["input_dim"], checkpoint["params"]["hidden_dims"], checkpoint["params"]["dropout"]).to(self.device_)
        self.model_.load_state_dict(checkpoint["state_dict"])
        self.model_.eval()
        self.preprocessor_ = joblib.load(ROOT / (preprocessor_path or selected["preprocessor_path"]))
        self.classes_ = np.array([0, 1])
        self.threshold_ = selected["threshold"]

    def fit(self, X, y):
        raise RuntimeError("This inference wrapper cannot fit; run python -m src.neural train.")

    @torch.inference_mode()
    def predict_proba(self, X):
        tensor = torch.from_numpy(self.preprocessor_.transform(X)).to(self.device_)
        scores = torch.sigmoid(self.model_(tensor)).cpu().numpy()
        return np.column_stack([1 - scores, scores])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.threshold_).astype(int)


def load_predictor():
    return NeuralPredictor()


def predict_scores(frame):
    return load_predictor().predict_proba(frame)[:, 1]


def write_notes(selected, history):
    best = history[selected["best_epoch"] - 1]
    last = history[-1]
    h1, h2 = selected["params"]["hidden_dims"]
    device_desc = f"GPU ({selected.get('gpu_device_name', 'CUDA')})" if selected.get("device") == "cuda" else "CPU"
    text = f"""# PyTorch MLP

The network uses {selected['input_dim']} transformed inputs, hidden layers {h1} and {h2}, ReLU after each hidden layer, dropout {selected['params']['dropout']}, and one output logit. Sigmoid is applied only for inference. Trainable parameters: {selected['parameter_count']:,}.

Training uses {device_desc}, seed {selected['seed']}, {selected['threads']} Torch threads, unweighted BCEWithLogitsLoss, and AdamW. Learning rate {selected['params']['learning_rate']}; weight decay {selected['params']['weight_decay']}; batch size {selected['batch_size']}. The budget is {selected['max_epochs']} epochs per candidate and early stopping after {selected['patience']} epochs without improved validation AP. There are two predeclared candidates in config.json. The selected candidate ran {selected['epochs_run']} epochs and restored epoch {selected['best_epoch']}. No final retraining on train+validation is done, so the validation threshold remains associated with the exact saved predictor.

Validation AP: {selected['validation_ap']:.6f}. Working threshold: {selected['threshold']:.6f}, chosen by maximum validation F1 with the highest-threshold tie break. These decisions never use test outcomes.

Inputs use the common preprocessing pipeline: signed log1p on monetary columns, training-fitted StandardScaler, and one-hot codes including undocumented categories kept separate. ID and target are excluded. No oversampling or class-weighted loss is used: AP evaluates ranking under the observed prevalence, while the validation threshold sets the precision/recall tradeoff. This avoids treating a weighted-loss sigmoid as an automatically calibrated probability. Even the unweighted sigmoid still requires separate calibration validation before operational use.

## Reading the learning curves

At the selected epoch, training AP is {best['train_ap']:.6f} and validation AP is {best['validation_ap']:.6f}; training loss is {best['train_loss']:.6f} and validation loss is {best['validation_loss']:.6f}. At the last epoch, training AP is {last['train_ap']:.6f} and validation AP is {last['validation_ap']:.6f}, with losses {last['train_loss']:.6f} and {last['validation_loss']:.6f}. A growing training/validation gap with no validation AP improvement is evidence of overfitting to this training sample; patience restores the earlier checkpoint. Small epoch-to-epoch AP changes are also affected by finite validation data. Both curves are measured in eval mode with dropout disabled, making their comparison consistent. This single seed and small search do not establish a universal ranking between architectures.

If the MLP trails boosting, compare the saved validation and test AP values. Trees can capture threshold effects and feature interactions efficiently on modest tabular datasets; this two-candidate MLP search may not find the best neural configuration. That is an explanation to investigate, not proof from one split.

## Backpropagation and artifact loading

The forward pass computes logits and binary cross-entropy. loss.backward() applies the chain rule through the layers to compute a gradient for each parameter. optimizer.step() uses AdamW to update those parameters and apply weight decay. zero_grad() clears old gradients before each minibatch. Validation runs without gradients and cannot update weights.

Code: src/neural.py::train_candidate. Run `python -m src.neural train`. Files include both candidate checkpoints, all epoch histories, selected.json, train-fitted preprocessor.joblib, and learning_curves.png. `load_predictor().predict_proba(raw_frame)[:, 1]` loads the exact selected checkpoint and returns positive-class scores. Saved checkpoint reload is checked against in-memory validation predictions with exact equality during training. Predictions are model probability estimates, not proven calibrated individual risks.
"""
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs/neural.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train"], nargs="?", default="train")
    parser.parse_args()
    train()


if __name__ == "__main__":
    main()
