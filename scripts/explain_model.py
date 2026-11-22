"""3.2 fairness, 3.4 global explainability, 3.5 reasoning for three cases."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from sklearn.metrics import f1_score, recall_score  # noqa: E402

from qa_model.config import MESH_FEATURES, REPORTS_DIR  # noqa: E402
from qa_model.explain import shap_values_for  # noqa: E402
from qa_model.predict import QAModel  # noqa: E402


def pick_cases(test: pd.DataFrame, pred: pd.Series) -> dict:
    """Three cases from the report: raw Tripo (FAIL), fixed by Modeler C (PASS), Mobile prop (FAIL)."""
    cases = {}
    raw = test[(test.source == "tripo_raw") & (pred == 1) & (test.tri_budget_ratio < 1)]
    cases["case1_tripo_raw"] = raw.sort_values("islands_per_1k_tris", ascending=False).index[0]
    fixed = test[(test.source == "tripo_fixed") & (pred == 0)]
    cases["case2_after_modeler_c"] = fixed.index[0]
    mobile = test[(test.platform == "Mobile") & (test.asset_role == "prop") & (pred == 1)
                  & (test.texture_budget_ratio > 1)]
    if mobile.empty:
        mobile = test[(test.platform == "Mobile") & (pred == 1)]
    cases["case3_mobile_prop"] = mobile.index[0]
    return cases


def main():
    model = QAModel()
    test = pd.read_csv(REPORTS_DIR / "test_split.csv")
    p = pd.Series(model.pipeline.predict_proba(test)[:, 1], index=test.index)
    pred = (p >= model.threshold).astype(int)

    # 3.2 fairness: equal opportunity = recall on FAIL per group
    rows = []
    for group in ("source", "platform"):
        for g, part in test.groupby(group):
            if part.label.sum() == 0:
                continue
            rows.append({"group": group, "value": g, "n": len(part),
                         "fail_rate": round(part.label.mean(), 3),
                         "recall_fail": round(recall_score(part.label, pred[part.index]), 3),
                         "f1_fail": round(f1_score(part.label, pred[part.index]), 3)})
    fair = pd.DataFrame(rows)
    fair.to_csv(REPORTS_DIR / "fairness.csv", index=False)
    src = fair[fair.group == "source"]
    print("== 3.2 recall on FAIL by group (equal opportunity)\n", fair.to_string(index=False))
    print(f"equal-opportunity gap across sources: {src.recall_fail.max() - src.recall_fail.min():.3f}")

    # 3.4 global: SHAP summary
    sv, Xt = shap_values_for(model.pipeline, test, model.explainer)
    plt.figure()
    shap.summary_plot(sv.values, Xt, show=False, max_display=14)
    plt.title(f"SHAP summary ({model.version}) - push toward FAIL > 0")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "shap_summary.png", dpi=120, bbox_inches="tight")
    plt.close()
    importance = pd.Series(abs(sv.values).mean(0), index=Xt.columns).sort_values(ascending=False)
    importance.round(4).to_csv(REPORTS_DIR / "shap_importance.csv", header=["mean_abs_shap"])
    print("\n== 3.4 mean |SHAP| (top 8)\n", importance.head(8).round(4).to_string())

    # 3.5 three individual decisions
    out = {}
    for name, idx in pick_cases(test, pred).items():
        row = test.loc[idx]
        result = model.assess({f: row[f] for f in MESH_FEATURES}, row.platform, row.asset_role)
        out[name] = {"asset_id": row.asset_id, "source": row.source, "platform": row.platform,
                     "asset_role": row.asset_role, "true_label": "FAIL" if row.label else "PASS",
                     **{k: result[k] for k in ("status", "readiness_score", "issues", "reasons", "next_action")}}
        i = test.index.get_loc(idx)
        plt.figure()
        shap.plots.waterfall(sv[i], max_display=10, show=False)
        plt.title(f"{name}: {result['status']} (score {result['readiness_score']})")
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / f"waterfall_{name}.png", dpi=120, bbox_inches="tight")
        plt.close()
        print(f"\n== 3.5 {name} [{row.source}, {row.platform}/{row.asset_role}] -> "
              f"{result['status']} score={result['readiness_score']}")
        for r in result["reasons"]:
            print(f"   {r['shap']:+.3f}  {r['feature']:<22} {r['text']}")
    with open(REPORTS_DIR / "cases.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print("\nreports saved to", REPORTS_DIR)


if __name__ == "__main__":
    main()
