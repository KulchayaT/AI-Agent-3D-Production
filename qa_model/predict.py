"""Load the trained model and assess one asset (used by the API and the agents)."""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from .config import ISSUES, MESH_FEATURES, MODEL_PATH, SPEC
from .explain import shap_values_for, top_reasons
from .features import add_context_features


class QAModel:
    def __init__(self, path: str | Path = MODEL_PATH):
        bundle = joblib.load(path)
        self.pipeline = bundle["pipeline"]
        self.version = bundle["version"]
        self.threshold = bundle["threshold"]
        self.use_context = bundle["use_context"]
        self._explainer = None

    @property
    def explainer(self):
        if self._explainer is None:
            clf = self.pipeline.named_steps["clf"]
            if not hasattr(clf, "coef_"):
                import shap
                self._explainer = shap.TreeExplainer(clf)
        return self._explainer

    @staticmethod
    def rule_issues(row: dict) -> list[dict]:
        issues = []
        for feat, (kind, limit) in SPEC.items():
            v = row[feat]
            bad = v > limit if kind == "max" else v < limit
            if bad:
                code, severity, agent, task = ISSUES[feat]
                issues.append({"code": code, "feature": feat, "severity": severity,
                               "value": round(float(v), 4), "limit": limit,
                               "agent": agent, "task": task})
        order = {"high": 0, "medium": 1, "low": 2}
        return sorted(issues, key=lambda i: order[i["severity"]])

    def assess(self, features: dict, platform: str, asset_role: str) -> dict:
        missing = [f for f in MESH_FEATURES if f not in features]
        if missing:
            raise KeyError(f"missing features: {missing}")
        row = add_context_features({f: features[f] for f in MESH_FEATURES}, platform, asset_role)
        X = pd.DataFrame([row])
        p_fail = float(self.pipeline.predict_proba(X)[0, 1])
        fail = p_fail >= self.threshold
        sv, Xt = shap_values_for(self.pipeline, X, self.explainer)
        reasons = top_reasons(sv.values[0], list(Xt.columns), X.iloc[0], fail)
        issues = self.rule_issues(row)
        if fail and issues:
            next_action = {"agent": issues[0]["agent"], "task": issues[0]["task"]}
        elif fail:
            next_action = {"agent": "technical_artist_human", "task": "manual_review"}
        else:
            next_action = {"agent": "technical_artist_a", "task": "import_to_unreal"}
        return {
            "status": "FAIL" if fail else "PASS",
            "readiness_score": round(100 * (1 - p_fail)),
            "fail_probability": round(p_fail, 4),
            "model_version": self.version,
            "issues": issues,
            "reasons": reasons,
            "next_action": next_action,
            # model and rules disagree -> a human should look (report: MCS-2 mitigation)
            "needs_human_review": bool(fail != bool(issues)) or 0.35 < p_fail < 0.65,
            "features": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()},
        }
