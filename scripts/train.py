"""3.1 + 3.3: train 4 model versions, track them in MLflow, keep the best one.

Selection rule: highest F1 on FAIL among versions whose cross-validated Recall(FAIL)
reaches 0.90 (report goal). The decision threshold is tuned on out-of-fold predictions.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, cohen_kappa_score,  # noqa: E402
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split  # noqa: E402

from qa_model.config import DATA_PATH, EXPERIMENT, MLFLOW_URI, MODEL_PATH, REPORTS_DIR  # noqa: E402
from qa_model.model import VERSIONS, build, feature_columns  # noqa: E402

RECALL_GOAL = 0.90
SEED = 42


def metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    return {
        "recall_fail": recall_score(y, pred),
        "precision_fail": precision_score(y, pred, zero_division=0),
        "f1_fail": f1_score(y, pred),
        "accuracy": accuracy_score(y, pred),
        "roc_auc": roc_auc_score(y, p),
        "kappa": cohen_kappa_score(y, pred),
    }


def pick_threshold(y, p):
    """Highest-F1 threshold that keeps recall >= goal (fallback 0.5)."""
    best_t, best_f1 = 0.5, -1
    for t in np.arange(0.15, 0.86, 0.01):
        pred = (p >= t).astype(int)
        if recall_score(y, pred) >= RECALL_GOAL:
            f = f1_score(y, pred)
            if f > best_f1:
                best_t, best_f1 = float(round(t, 2)), f
    return best_t


def main():
    df = pd.read_csv(DATA_PATH)
    y = df["label"].values
    train_df, test_df, y_train, y_test = train_test_split(
        df, y, test_size=0.2, stratify=y, random_state=SEED)
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)
    REPORTS_DIR.mkdir(exist_ok=True)
    rows, fitted = [], {}
    for version in VERSIONS:
        pipe, params, use_ctx = build(version, pos_weight=pos_weight, seed=SEED)
        num, cat = feature_columns(use_ctx)
        cols = num + cat
        with mlflow.start_run(run_name=version):
            oof = cross_val_predict(pipe, train_df[cols], y_train, cv=cv, method="predict_proba")[:, 1]
            thr = pick_threshold(y_train, oof)
            cv_m = metrics(y_train, oof, thr)
            pipe.fit(train_df[cols], y_train)
            p_test = pipe.predict_proba(test_df[cols])[:, 1]
            test_m = metrics(y_test, p_test, thr)

            mlflow.log_params({**params, "version": version, "use_context": use_ctx,
                               "n_features": len(cols), "threshold": thr, "train_rows": len(train_df)})
            mlflow.log_metrics({f"cv_{k}": v for k, v in cv_m.items()})
            mlflow.log_metrics({f"test_{k}": v for k, v in test_m.items()})
            fig, ax = plt.subplots(figsize=(4, 4))
            ConfusionMatrixDisplay.from_predictions(
                y_test, (p_test >= thr).astype(int), display_labels=["PASS", "FAIL"], ax=ax, colorbar=False)
            ax.set_title(version)
            fig.tight_layout()
            cm_path = REPORTS_DIR / f"confusion_{version}.png"
            fig.savefig(cm_path, dpi=110)
            plt.close(fig)
            mlflow.log_artifact(str(cm_path))

        fitted[version] = (pipe, thr, use_ctx)
        rows.append({"version": version, "threshold": thr,
                     **{f"cv_{k}": round(v, 4) for k, v in cv_m.items()},
                     **{f"test_{k}": round(v, 4) for k, v in test_m.items()}})
        print(f"{version:22s} thr={thr:.2f}  CV recall={cv_m['recall_fail']:.3f} F1={cv_m['f1_fail']:.3f}"
              f" | TEST recall={test_m['recall_fail']:.3f} F1={test_m['f1_fail']:.3f}"
              f" AUC={test_m['roc_auc']:.3f} kappa={test_m['kappa']:.3f}")

    table = pd.DataFrame(rows)
    table.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    eligible = table[table.cv_recall_fail >= RECALL_GOAL]
    best = (eligible if len(eligible) else table).sort_values("cv_f1_fail", ascending=False).iloc[0]
    version = best["version"]
    pipe, thr, use_ctx = fitted[version]
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"pipeline": pipe, "version": version, "threshold": thr, "use_context": use_ctx},
                MODEL_PATH)
    test_df.to_csv(REPORTS_DIR / "test_split.csv", index=False)
    with open(REPORTS_DIR / "selected_model.json", "w", encoding="utf-8") as f:
        json.dump(best.to_dict(), f, indent=2)
    print(f"\nselected: {version} (threshold {thr}) -> {MODEL_PATH}")
    print(f"MLflow UI: mlflow ui --backend-store-uri {MLFLOW_URI}")


if __name__ == "__main__":
    main()
