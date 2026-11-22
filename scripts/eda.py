"""3.1 Basic data exploration: types, statistics, class balance, plots in reports/."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from qa_model.config import DATA_PATH, MESH_FEATURES, REPORTS_DIR  # noqa: E402

if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    REPORTS_DIR.mkdir(exist_ok=True)
    print("== shape:", df.shape)
    print("\n== dtypes\n", df.dtypes.to_string())
    print("\n== missing values:", int(df.isna().sum().sum()))
    print("\n== statistics\n", df[MESH_FEATURES].describe().T.round(3).to_string())
    print("\n== class distribution (1 = FAIL)\n", df["label"].value_counts().to_string())
    by_source = df.groupby("source")["label"].agg(["count", "mean"]).rename(columns={"mean": "fail_rate"})
    print("\n== fail rate by source\n", by_source.round(3).to_string())
    print("\n== fail rate by platform\n", df.groupby("platform")["label"].mean().round(3).to_string())
    corr = df[MESH_FEATURES + ["label"]].corr()["label"].drop("label").sort_values()
    print("\n== correlation with FAIL\n", corr.round(3).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    by_source["fail_rate"].plot.bar(ax=axes[0], color="#E0703A", title="FAIL rate by source")
    corr.plot.barh(ax=axes[1], color="#2F5FC4", title="Correlation with FAIL")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "eda_overview.png", dpi=120)

    feats = ["islands_per_1k_tris", "uv_overlap_pct", "texel_density_cv", "seam_ratio"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    for ax, f in zip(axes, feats):
        for lab, color in ((0, "#2F5FC4"), (1, "#E0703A")):
            df.loc[df.label == lab, f].clip(upper=df[f].quantile(0.98)).plot.hist(
                ax=ax, bins=30, alpha=0.6, color=color, label="FAIL" if lab else "PASS")
        ax.set_title(f)
        ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "eda_distributions.png", dpi=120)
    print("\nplots saved to", REPORTS_DIR)
