"""Download and audit UCI 350; statistical EDA uses the training partition only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL = "https://archive.ics.uci.edu/static/public/350/default%2Bof%2Bcredit%2Bcard%2Bclients.zip"
DATA_PATH = ROOT / "data/credit_default.csv"
PAY_COLUMNS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
FEATURES = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"] + PAY_COLUMNS + [f"BILL_AMT{i}" for i in range(1, 7)] + [f"PAY_AMT{i}" for i in range(1, 7)]


def validate_schema(df: pd.DataFrame) -> None:
    expected = ["ID"] + FEATURES + ["default"]
    if list(df.columns) != expected or len(df) != 30000:
        raise ValueError(f"Unexpected dataset shape/columns: {df.shape}, {list(df.columns)}")
    if not df.ID.is_unique or set(df.default.unique()) != {0, 1}:
        raise ValueError("Invalid IDs or target values")
    if df.isna().any().any():
        raise ValueError("Unexpected missing values in the published dataset")
    if not np.isfinite(df.to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite data")


def prepare_data() -> pd.DataFrame:
    """Cache official ZIP/XLS and create an integer CSV without changing row order."""
    raw = ROOT / "data/raw"
    raw.mkdir(parents=True, exist_ok=True)
    archive = raw / "uci_credit_default.zip"
    if not archive.exists():
        request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "ML-final-project/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read()
        archive.write_bytes(payload)
    with zipfile.ZipFile(archive) as zf:
        names = [name for name in zf.namelist() if name.lower().endswith(".xls")]
        if len(names) != 1:
            raise ValueError(f"Expected one XLS in archive, found {names}")
        xls_path = raw / Path(names[0]).name
        xls_path.write_bytes(zf.read(names[0]))
    df = pd.read_excel(xls_path, header=1, engine="xlrd")
    df.columns = [str(name).strip() for name in df.columns]
    df = df.rename(columns={"default payment next month": "default"}).astype("int64")
    validate_schema(df)
    df.to_csv(DATA_PATH, index=False, lineterminator="\n")
    provenance = {
        "source_url": SOURCE_URL,
        "dataset_page": "https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients",
        "citation": "Yeh, I. (2009). Default of Credit Card Clients. UCI. DOI:10.24432/C55S3H",
        "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "xls_sha256": hashlib.sha256(xls_path.read_bytes()).hexdigest(),
        "csv_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
        "changes": "Read XLS header row 2, rename target to default, integer CSV; preserve all rows and codes.",
        "rows": len(df), "predictors": len(FEATURES),
    }
    (raw / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"Prepared {DATA_PATH}: {len(df)} rows, {len(FEATURES)} predictors")
    return df


def load_data() -> pd.DataFrame:
    """Load the canonical raw frame; never silently download during modeling."""
    if not DATA_PATH.exists():
        raise FileNotFoundError("Run python -m src.data download first")
    df = pd.read_csv(DATA_PATH)
    validate_schema(df)
    return df


def structural_audit(df: pd.DataFrame) -> dict:
    """Whole-dataset schema and duplicate audit; no predictive model statistics."""
    groups = df.groupby(FEATURES, dropna=False, sort=False)["default"].agg(["size", "nunique"])
    duplicate_groups = groups[groups["size"] > 1]
    conflicting = duplicate_groups[duplicate_groups["nunique"] > 1]
    return {
        "rows": len(df), "predictor_count": len(FEATURES),
        "missing_cells": int(df.isna().sum().sum()),
        "unique_ids": int(df.ID.nunique()),
        "duplicate_rows_excluding_id_extra": int(df.drop(columns="ID").duplicated().sum()),
        "duplicate_feature_rows_extra": int(df[FEATURES].duplicated().sum()),
        "feature_duplicate_groups": len(duplicate_groups),
        "rows_in_feature_duplicate_groups": int(duplicate_groups["size"].sum()),
        "conflicting_target_groups": len(conflicting),
        "rows_in_conflicting_target_groups": int(conflicting["size"].sum()),
        "categorical_code_counts": {col: {str(k): int(v) for k, v in df[col].value_counts().sort_index().items()} for col in ["SEX", "EDUCATION", "MARRIAGE"] + PAY_COLUMNS},
        "duplicate_policy": "Keep all rows; assign identical raw feature vectors to the same partition, including conflicting outcomes. ID and target excluded from grouping.",
        "scope": "Full-data structural audit only; distribution and outcome EDA restricted to train.",
    }


def run_eda() -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.preprocessing import load_splits

    df = load_data()
    indices = load_splits()["train"]
    train = df.iloc[indices].copy()
    out = ROOT / "artifacts/eda"
    out.mkdir(parents=True, exist_ok=True)
    audit = structural_audit(df)
    (out / "structural_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    summary = {"scope": "train only", "n_train": len(train), "positive_count": int(train.default.sum()), "positive_fraction": float(train.default.mean()), "zero_payment_fraction": {c: float(train[c].eq(0).mean()) for c in [f"PAY_AMT{i}" for i in range(1, 7)]}, "negative_bill_counts": {c: int(train[c].lt(0).sum()) for c in [f"BILL_AMT{i}" for i in range(1, 7)]}}
    (out / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    train[FEATURES].describe(percentiles=[.01,.25,.5,.75,.99]).T.to_csv(out / "train_descriptive_statistics.csv")
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7, 4))
    counts = train.default.value_counts().reindex([0,1])
    bars = ax.bar(["No default", "Default"], counts, color=["#9AA9B7", "#176B87"])
    ax.bar_label(bars, labels=[f"{v:,} ({v/len(train):.1%})" for v in counts], padding=4)
    ax.set(title="Training class balance", ylabel="Clients", ylim=(0, max(counts)*1.15))
    fig.tight_layout(); fig.savefig(out / "class_balance.png", dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, col in zip(axes, ["LIMIT_BAL", "BILL_AMT1", "PAY_AMT1"]):
        vals = train[col]
        transformed = np.sign(vals)*np.log10(1+np.abs(vals))
        ax.hist(transformed, bins=45, color="#176B87", alpha=.9)
        ax.set(title=col, xlabel="sign(x) × log10(1 + |x / NT$|)", ylabel="Clients")
    fig.suptitle("Training monetary distributions: signed-log display preserves negatives")
    fig.tight_layout(); fig.savefig(out / "amount_distributions.png", dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    pay_counts = train.PAY_0.value_counts().sort_index()
    axes[0].bar(pay_counts.index.astype(str), pay_counts, color="#176B87")
    axes[0].set(title="September payment-status codes", xlabel="PAY_0 (−2 and 0 undocumented by UCI)", ylabel="Training clients")
    rates = train.groupby("PAY_0").default.agg(["mean", "size"])
    axes[1].bar(rates.index.astype(str), rates["mean"], color="#176B87")
    axes[1].set(title="Default fraction by September code", xlabel="PAY_0 (small groups are unstable)", ylabel="Training default fraction", ylim=(0,1))
    fig.tight_layout(); fig.savefig(out / "payment_status.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 6))
    cols = PAY_COLUMNS + ["LIMIT_BAL", "BILL_AMT1", "PAY_AMT1", "default"]
    corr = train[cols].corr(method="spearman")
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(cols)), cols, rotation=60, ha="right")
    ax.set_yticks(range(len(cols)), cols)
    ax.set_title("Training Spearman correlations")
    fig.colorbar(im, ax=ax, label="Spearman correlation")
    fig.tight_layout(); fig.savefig(out / "correlations.png", dpi=160); plt.close(fig)
    rates.to_csv(out / "train_pay0_default_rates.csv")
    print(json.dumps({"audit": audit, "train_summary": summary}, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["download", "eda", "audit"])
    args = parser.parse_args()
    if args.command == "download":
        prepare_data()
    elif args.command == "eda":
        run_eda()
    else:
        print(json.dumps(structural_audit(load_data()), indent=2))


if __name__ == "__main__":
    main()
