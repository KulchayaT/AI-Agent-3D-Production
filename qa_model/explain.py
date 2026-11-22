"""Turn model features and SHAP values into reasons a Technical Artist can read."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SPEC

# Text when a feature pushes the prediction toward FAIL / toward PASS.
REASON_TEXT = {
    "islands_per_1k_tris": ("UV แตกเป็นชิ้นเล็กมากเกินไป", "จำนวน UV island เหมาะสม"),
    "uv_island_count": ("จำนวน UV island สูง", "จำนวน UV island ต่ำ"),
    "uv_overlap_pct": ("UV ทับกัน (overlap)", "UV ไม่ทับกัน"),
    "texel_density_cv": ("ความละเอียด Texture ไม่เท่ากันทั้งตัว", "Texel density สม่ำเสมอ"),
    "uv_stretch": ("UV ยืดหรือบิดเบี้ยว", "UV ไม่ยืด"),
    "seam_ratio": ("มี Seam มากเกินไป", "จำนวน Seam เหมาะสม"),
    "uv_coverage": ("ใช้พื้นที่ UV น้อย เสียพื้นที่ Texture", "ใช้พื้นที่ UV ได้ดี"),
    "non_manifold_edges": ("มี Non-manifold edge", "Mesh สะอาด ไม่มี Non-manifold"),
    "material_slots": ("Material slot มากเกิน (Draw call สูง)", "จำนวน Material เหมาะสม"),
    "tri_budget_ratio": ("Triangle เกิน Budget ของ Platform", "Triangle อยู่ใน Budget"),
    "texture_budget_ratio": ("Texture ใหญ่เกิน Budget ของ Platform", "ขนาด Texture อยู่ใน Budget"),
    "tri_count": ("จำนวน Triangle สูง", "จำนวน Triangle ต่ำ"),
    "texture_res": ("Texture ความละเอียดสูง", "Texture ความละเอียดพอดี"),
    "quad_ratio": ("Topology ไม่ใช่ Quad เป็นส่วนใหญ่", "Topology เป็น Quad"),
}


def base_feature(name: str) -> str:
    """'platform_Mobile' -> 'platform' (one-hot columns)."""
    for prefix in ("platform", "asset_role"):
        if name.startswith(prefix + "_"):
            return prefix
    return name


def reason_text(feature: str, shap_value: float, raw_value) -> str:
    base = base_feature(feature)
    if base in ("platform", "asset_role"):
        value = feature.split("_", 1)[1] if base == "platform" else feature[len("asset_role_"):]
        return f"Context: {base} = {value}"
    fail_txt, pass_txt = REASON_TEXT.get(base, (base, base))
    if shap_value > 0 and base in SPEC and raw_value is not None and not isinstance(raw_value, str):
        kind, limit = SPEC[base]
        broken = raw_value > limit if kind == "max" else raw_value < limit
        if not broken:  # pushes toward FAIL but still inside the SPEC
            return f"{pass_txt} ({float(raw_value):.2f} / เกณฑ์ {limit}) แต่ยังดันผลไปทาง FAIL"
    return fail_txt if shap_value > 0 else pass_txt


def tree_explainer(pipeline):
    import shap

    return shap.TreeExplainer(pipeline.named_steps["clf"])


def shap_values_for(pipeline, X: pd.DataFrame, explainer=None):
    """SHAP values (log-odds of FAIL) on the transformed feature space."""
    import shap

    prep, clf = pipeline.named_steps["prep"], pipeline.named_steps["clf"]
    Xt = pd.DataFrame(prep.transform(X), columns=prep.get_feature_names_out(), index=X.index)
    if explainer is None:
        if hasattr(clf, "coef_"):
            explainer = shap.LinearExplainer(clf, Xt)
        else:
            explainer = shap.TreeExplainer(clf)
    sv = explainer(Xt)
    values = sv.values
    if values.ndim == 3:  # random forest: (n, features, classes)
        values = values[:, :, 1]
        base = np.atleast_1d(sv.base_values)[..., 1] if np.ndim(sv.base_values) > 1 else sv.base_values
        sv = shap.Explanation(values, base_values=base, data=sv.data, feature_names=sv.feature_names)
    return sv, Xt


def top_reasons(shap_row: np.ndarray, feature_names, raw_row: pd.Series, status_fail: bool, k: int = 3):
    """k features that pushed hardest toward the predicted status."""
    order = np.argsort(-shap_row) if status_fail else np.argsort(shap_row)
    out = []
    for i in order[:k]:
        name = feature_names[i]
        val = float(shap_row[i])
        if (status_fail and val <= 0) or (not status_fail and val >= 0):
            continue
        base = base_feature(name)
        raw = raw_row.get(base) if base in raw_row else None
        out.append({
            "feature": name,
            "value": None if raw is None or isinstance(raw, str) else round(float(raw), 4),
            "shap": round(val, 4),
            "text": reason_text(name, val, raw),
        })
    return out
