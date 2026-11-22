"""Model versions compared in MLflow (report 3.3)."""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .config import CONTEXT_CAT_FEATURES, CONTEXT_NUM_FEATURES, MESH_FEATURES


def feature_columns(use_context: bool) -> tuple[list[str], list[str]]:
    num = MESH_FEATURES + (CONTEXT_NUM_FEATURES if use_context else [])
    cat = CONTEXT_CAT_FEATURES if use_context else []
    return num, cat


def make_preprocessor(use_context: bool, scale: bool) -> ColumnTransformer:
    num, cat = feature_columns(use_context)
    parts = [("num", StandardScaler() if scale else "passthrough", num)]
    if cat:
        parts.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat))
    return ColumnTransformer(parts, verbose_feature_names_out=False)


def build(version: str, pos_weight: float = 1.0, seed: int = 42) -> tuple[Pipeline, dict, bool]:
    """Return (pipeline, params to log, uses_context)."""
    if version == "v1_logreg":
        params = {"C": 1.0, "class_weight": "balanced", "max_iter": 2000}
        clf = LogisticRegression(**params)
        return Pipeline([("prep", make_preprocessor(False, True)), ("clf", clf)]), params, False
    if version == "v2_random_forest":
        params = {"n_estimators": 300, "max_depth": 10, "min_samples_leaf": 2,
                  "class_weight": "balanced", "random_state": seed}
        clf = RandomForestClassifier(n_jobs=-1, **params)
        return Pipeline([("prep", make_preprocessor(False, False)), ("clf", clf)]), params, False
    if version == "v3_xgboost":
        params = {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.08,
                  "subsample": 0.9, "colsample_bytree": 0.9, "random_state": seed}
        clf = XGBClassifier(eval_metric="logloss", **params)
        return Pipeline([("prep", make_preprocessor(False, False)), ("clf", clf)]), params, False
    if version == "v4_xgboost_context":
        params = {"n_estimators": 400, "max_depth": 4, "learning_rate": 0.06,
                  "subsample": 0.9, "colsample_bytree": 0.9, "min_child_weight": 2,
                  "scale_pos_weight": round(pos_weight, 3), "random_state": seed}
        clf = XGBClassifier(eval_metric="logloss", **params)
        return Pipeline([("prep", make_preprocessor(True, False)), ("clf", clf)]), params, True
    raise ValueError(version)


VERSIONS = ["v1_logreg", "v2_random_forest", "v3_xgboost", "v4_xgboost_context"]
