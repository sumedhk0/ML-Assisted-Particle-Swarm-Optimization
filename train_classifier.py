"""Train the LightGBM stuck-particle classifier.

Splits by run (not row) to avoid leakage. Prints test AUC/F1 and the top-5
feature importances so you can sanity-check that pbest-plateau / value-rank
dominate — that's a strong signal the labels are meaningful.
"""
import argparse

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from features import FEATURE_NAMES
from stuck_classifier import StuckClassifier


def split_by_run(run_ids: np.ndarray, seed: int = 42):
    unique = np.unique(run_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_train = int(0.8 * len(unique))
    n_val = int(0.1 * len(unique))
    return (set(unique[:n_train]),
            set(unique[n_train:n_train + n_val]),
            set(unique[n_train + n_val:]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="training_data.npz")
    parser.add_argument("--out", type=str, default="stuck_classifier.lgb")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    data = np.load(args.data)
    X = data["X"]
    y = data["y"]
    run_ids = data["run_ids"]

    train_runs, val_runs, test_runs = split_by_run(run_ids)
    tr = np.isin(run_ids, list(train_runs))
    va = np.isin(run_ids, list(val_runs))
    te = np.isin(run_ids, list(test_runs))

    print(f"rows: train={tr.sum():,}  val={va.sum():,}  test={te.sum():,}")
    print(f"pos rate: train={y[tr].mean():.3f}  val={y[va].mean():.3f}  test={y[te].mean():.3f}")

    clf = StuckClassifier(device=args.device)
    clf.fit(X[tr], y[tr], X[va], y[va])

    preds = clf.predict_proba(X[te])
    auc = roc_auc_score(y[te], preds)
    f1 = f1_score(y[te], preds > 0.5)
    print(f"\nTEST  AUC={auc:.4f}  F1={f1:.4f}")

    imp = clf.booster.feature_importance(importance_type="gain")
    order = np.argsort(-imp)
    print("\nTop features by gain:")
    for i in order[:5]:
        print(f"  {FEATURE_NAMES[i]:20s} {imp[i]:.1f}")

    clf.save(args.out)
    print(f"\nsaved to {args.out}")


if __name__ == "__main__":
    main()
