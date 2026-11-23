"""One frozen client split and train-fitted preprocessing shared by every model.

Inspect ``create_splits`` during the defense for duplicate leakage prevention.
Inspect ``make_preprocessor`` to see exactly which transformations learn values.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "artifacts" / "splits"
DATA_PATH = ROOT / "data" / "credit_default.csv"
SEED = 42
TARGET = "default"
CATEGORICAL = ["SEX", "EDUCATION", "MARRIAGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
NUMERIC = ["LIMIT_BAL", "AGE"] + [f"BILL_AMT{i}" for i in range(1, 7)] + [f"PAY_AMT{i}" for i in range(1, 7)]
FEATURES = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"] + [f"BILL_AMT{i}" for i in range(1, 7)] + [f"PAY_AMT{i}" for i in range(1, 7)]


def _read_data() -> pd.DataFrame:
    from src.data import load_data
    frame = load_data()
    assert len(frame) == 30000, "Expected the complete 30,000-client UCI data."
    assert set(frame.columns) == set(FEATURES + ["ID", TARGET]), "Unexpected schema."
    assert frame["ID"].is_unique, "ID must uniquely identify source records."
    assert not frame.isna().any().any(), "Unexpected missing data; review before training."
    return frame.reset_index(drop=True)


def source_sha256() -> str:
    return hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()


def feature_groups(frame: pd.DataFrame) -> np.ndarray:
    """Exact predictor equality, independent of ID AND outcome.

    MultiIndex.factorize compares complete row values (rather than trusting a
    hash collision check). Contradictory outcomes remain in the same group.
    """
    return pd.MultiIndex.from_frame(frame[FEATURES]).factorize(sort=False)[0]


def check_splits(frame: pd.DataFrame, splits: dict[str, np.ndarray]) -> dict:
    groups = feature_groups(frame)
    names = ["train", "validation", "test"]
    joined = np.concatenate([splits[n] for n in names])
    assert np.array_equal(np.sort(joined), np.arange(len(frame))), "Splits must cover all rows exactly once."
    audit = {"seed": SEED, "n_rows": len(frame), "n_feature_groups": int(len(np.unique(groups))), "splits": {}}
    for n in names:
        ix = splits[n]
        y = frame.iloc[ix][TARGET]
        audit["splits"][n] = {"n": len(ix), "positive_count": int(y.sum()), "positive_rate": float(y.mean()), "fraction": len(ix) / len(frame)}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not np.intersect1d(splits[a], splits[b]).size, "Row overlap."
            assert not np.intersect1d(groups[splits[a]], groups[splits[b]]).size, "Duplicate feature group leakage."
    audit["row_overlap"] = 0
    audit["feature_group_overlap"] = 0
    audit["id_in_features"] = False
    audit["target_in_features"] = False
    group_targets = pd.DataFrame({"group": groups, "y": frame[TARGET]})
    audit["groups_with_conflicting_targets"] = int((group_targets.groupby("group")["y"].nunique() > 1).sum())
    audit["duplicate_feature_rows_beyond_first"] = int(frame[FEATURES].duplicated().sum())
    audit["duplicate_rows_without_id_beyond_first"] = int(frame[FEATURES + [TARGET]].duplicated().sum())
    return audit


def create_splits(seed: int = SEED, force: bool = False) -> dict[str, np.ndarray]:
    """Freeze approximately stratified 70/15/15, keeping duplicates together.

    Twenty disjoint folds allow a fixed 14/3/3 assignment without searching
    seeds or optimizing the split based on downstream model results.
    """
    if seed != SEED:
        raise ValueError("This experiment's predeclared seed is 42.")
    if (SPLIT_DIR / "indices.npz").exists() and not force:
        return load_splits()
    frame = _read_data()
    groups = feature_groups(frame)
    splitter = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=seed)
    folds = [held_out for _, held_out in splitter.split(frame[FEATURES], frame[TARGET], groups)]
    splits = {
        "train": np.sort(np.concatenate(folds[:14])),
        "validation": np.sort(np.concatenate(folds[14:17])),
        "test": np.sort(np.concatenate(folds[17:])),
    }
    audit = check_splits(frame, splits)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(SPLIT_DIR / "indices.npz", **splits)
    (SPLIT_DIR / "indices.json").write_text(json.dumps({k: v.tolist() for k, v in splits.items()}), encoding="utf-8")
    manifest = {
        "seed": seed,
        "source_sha256": source_sha256(),
        "method": "StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=42); folds 0:14 train, 14:17 validation, 17:20 test",
        "group_definition": "Exact equality of all 23 original predictors, excluding ID and target",
        "primary_selection_metric": "validation average_precision_score",
        "threshold_rule": "maximize validation F1 over distinct validation prediction scores; break exact ties with highest threshold; predict positive for p >= threshold",
        "test_policy": "No test input to fitting, early stopping, configuration selection, model selection, or threshold selection. Final report only after decisions are frozen.",
        "audit": audit,
    }
    (SPLIT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return splits


def load_splits() -> dict[str, np.ndarray]:
    manifest = json.loads((SPLIT_DIR / "manifest.json").read_text(encoding="utf-8"))
    if manifest["source_sha256"] != source_sha256():
        raise ValueError("Data fingerprint changed; frozen indices cannot be reused.")
    if manifest["seed"] != SEED:
        raise ValueError("Unexpected split seed.")
    with np.load(SPLIT_DIR / "indices.npz", allow_pickle=False) as archive:
        splits = {name: archive[name].copy() for name in ["train", "validation", "test"]}
    check_splits(_read_data(), splits)
    return splits


def load_partitions(include_test: bool = False) -> dict[str, tuple[pd.DataFrame, np.ndarray]]:
    """Return training/validation only unless final evaluation explicitly opts in."""
    frame = _read_data()
    splits = load_splits()
    names = ["train", "validation", "test"] if include_test else ["train", "validation"]
    return {n: (frame.iloc[splits[n]][FEATURES].copy(), frame.iloc[splits[n]][TARGET].to_numpy(dtype=np.int64)) for n in names}


def signed_log_numeric(values) -> np.ndarray:
    """Fixed, nonlearned transform: signed log1p money; preserve AGE in years."""
    values = np.asarray(values, dtype=np.float64).copy()
    money_columns = [i for i, name in enumerate(NUMERIC) if name != "AGE"]
    values[:, money_columns] = np.sign(values[:, money_columns]) * np.log1p(np.abs(values[:, money_columns]))
    return values


def as_float32(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def make_preprocessor() -> Pipeline:
    """Call fit/fit_transform ONLY on X_train; transform validation/test.

    Categorical codes remain separate, including codes UCI does not define.
    Numeric scaling learns means and variances on train only. No imputation is
    needed because structural validation rejects unexpected missing values.
    """
    numeric = Pipeline([
        ("signed_log", FunctionTransformer(signed_log_numeric, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    columns = ColumnTransformer([
        ("numeric", numeric, NUMERIC),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32), CATEGORICAL),
    ], remainder="drop", sparse_threshold=0)
    return Pipeline([
        ("columns", columns),
        ("float32", FunctionTransformer(as_float32, feature_names_out="one-to-one")),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "check"], nargs="?", default="prepare")
    args = parser.parse_args()
    splits = create_splits() if args.command == "prepare" else load_splits()
    print(json.dumps(check_splits(_read_data(), splits), indent=2))


if __name__ == "__main__":
    main()
