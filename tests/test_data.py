from qa_model.config import CONTEXT_NUM_FEATURES, MESH_FEATURES, SOURCES
from qa_model.data import make_dataset, rule_violations


def test_dataset_shape_and_columns():
    df = make_dataset(n_per_source=40, seed=1)
    assert len(df) == 40 * len(SOURCES)
    for col in MESH_FEATURES + CONTEXT_NUM_FEATURES + ["platform", "asset_role", "label", "source"]:
        assert col in df.columns
    assert df[MESH_FEATURES].isna().sum().sum() == 0
    assert set(df.label.unique()) <= {0, 1}


def test_raw_tripo_fails_more_than_fixed():
    df = make_dataset(n_per_source=200, seed=2)
    rates = df.groupby("source").label.mean()
    assert rates["tripo_raw"] > rates["tripo_fixed"] + 0.3


def test_rule_violation_on_overlap():
    row = {f: 0 for f in MESH_FEATURES + CONTEXT_NUM_FEATURES}
    row.update({"uv_coverage": 0.8, "uv_overlap_pct": 5.0})
    assert rule_violations(row) == ["uv_overlap_pct"]
