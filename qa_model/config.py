"""Shared constants: feature lists, studio budgets and SPEC thresholds.

The thresholds are starting values from the report (2.2.2 SPEC). They are meant to be
tuned with a Technical Artist once real Tripo outputs are collected.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "assets.csv"
MODEL_PATH = ROOT / "models" / "qa_model.joblib"
REPORTS_DIR = ROOT / "reports"
MLFLOW_URI = f"sqlite:///{(ROOT / 'mlruns.db').as_posix()}"
EXPERIMENT = "uv-readiness-qa"

PLATFORMS = ["PC", "Console", "Mobile"]
ASSET_ROLES = ["hero_character", "npc_character", "prop"]
SOURCES = ["tripo_raw", "tripo_fixed", "artist", "objaverse"]

# Triangle budget per platform and asset role.
TRI_BUDGET = {
    "PC": {"hero_character": 50_000, "npc_character": 20_000, "prop": 8_000},
    "Console": {"hero_character": 40_000, "npc_character": 15_000, "prop": 6_000},
    "Mobile": {"hero_character": 15_000, "npc_character": 6_000, "prop": 2_000},
}
# Largest texture side allowed per platform (px).
TEXTURE_BUDGET = {"PC": 4096, "Console": 4096, "Mobile": 2048}

# Features measured on the mesh itself (no context).
MESH_FEATURES = [
    "tri_count",
    "quad_ratio",
    "non_manifold_edges",
    "uv_island_count",
    "islands_per_1k_tris",
    "uv_overlap_pct",
    "uv_stretch",
    "texel_density_cv",
    "seam_ratio",
    "uv_coverage",
    "material_slots",
    "texture_res",
]
# Features that only make sense with the target platform / asset role.
CONTEXT_NUM_FEATURES = ["tri_budget_ratio", "texture_budget_ratio"]
CONTEXT_CAT_FEATURES = ["platform", "asset_role"]

# SPEC thresholds. ("max", x) = value must be <= x, ("min", x) = value must be >= x.
SPEC = {
    "uv_overlap_pct": ("max", 1.0),
    "texel_density_cv": ("max", 0.25),
    "uv_stretch": ("max", 0.15),
    "islands_per_1k_tris": ("max", 8.0),
    "seam_ratio": ("max", 0.15),
    "uv_coverage": ("min", 0.50),
    "non_manifold_edges": ("max", 0),
    "material_slots": ("max", 3),
    "tri_budget_ratio": ("max", 1.0),
    "texture_budget_ratio": ("max", 1.0),
}

# Issue code, severity and the agent that should fix it, per SPEC feature.
ISSUES = {
    "uv_overlap_pct": ("UV_OVERLAP", "high", "3d_modeler_c", "re_unwrap"),
    "texel_density_cv": ("TEXEL_DENSITY_UNEVEN", "medium", "3d_modeler_c", "re_pack_uv"),
    "uv_stretch": ("UV_STRETCH", "medium", "3d_modeler_c", "re_unwrap"),
    "islands_per_1k_tris": ("UV_FRAGMENTED", "high", "3d_modeler_c", "re_unwrap"),
    "seam_ratio": ("TOO_MANY_SEAMS", "medium", "3d_modeler_c", "re_unwrap"),
    "uv_coverage": ("UV_SPACE_WASTED", "low", "3d_modeler_c", "re_pack_uv"),
    "non_manifold_edges": ("NON_MANIFOLD", "high", "3d_modeler_b", "remesh"),
    "material_slots": ("TOO_MANY_MATERIALS", "medium", "technical_artist_a", "merge_materials"),
    "tri_budget_ratio": ("OVER_TRI_BUDGET", "high", "technical_artist_a", "reduce_polycount"),
    "texture_budget_ratio": ("OVER_TEXTURE_BUDGET", "medium", "technical_artist_a", "downscale_textures"),
}
