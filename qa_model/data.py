"""Synthetic training data for the proof of concept.

Real labels need a Technical Artist to review hundreds of models, which is IS work.
Until then we simulate four sources whose feature ranges follow what we see in practice:

    tripo_raw    Tripo output: good quad topology, fragmented UVs with many seams
    tripo_fixed  the same meshes after 3D Modeler C re-unwraps and re-packs the UVs
    artist       hand-made assets: clean UVs, sometimes over budget
    objaverse    downloaded assets: very mixed quality

Labels come from the SPEC rules, with TA disagreement simulated as label noise that is
more likely near a threshold. Replace this with real extracted features + TA labels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ASSET_ROLES, PLATFORMS, SPEC, TEXTURE_BUDGET, TRI_BUDGET


def _clip(x, lo, hi):
    return np.clip(x, lo, hi)


def _source_block(rng: np.random.Generator, source: str, n: int) -> pd.DataFrame:
    r = rng
    platform = r.choice(PLATFORMS, n, p=[0.5, 0.25, 0.25])
    role = r.choice(ASSET_ROLES, n, p=[0.4, 0.3, 0.3])
    # triangle count is made for the asset role (sized around the PC budget of that role)
    role_budget = np.array([TRI_BUDGET["PC"][a] for a in role])
    tri_spread = {"tripo_raw": (0.55, 0.4), "tripo_fixed": (0.55, 0.4),
                  "artist": (0.5, 0.45), "objaverse": (0.7, 0.9)}[source]
    tri = np.round(role_budget * r.lognormal(np.log(tri_spread[0]), tri_spread[1], n)).astype(int)
    if source in ("tripo_raw", "tripo_fixed"):
        quad = r.uniform(0.88, 0.99, n)
        nonman = np.where(r.random(n) < 0.04, r.integers(1, 6, n), 0)
        mats = np.ones(n, dtype=int)
        tex = r.choice([1024, 2048, 4096], n, p=[0.15, 0.7, 0.15])
        if source == "tripo_raw":
            islands_k = r.lognormal(np.log(12), 0.5, n)
            overlap = np.abs(r.normal(1.4, 1.3, n))
            stretch = r.normal(0.13, 0.05, n)
            cv = r.normal(0.33, 0.10, n)
            seam = r.normal(0.19, 0.06, n)
            cover = r.normal(0.55, 0.10, n)
        else:
            islands_k = r.lognormal(np.log(3), 0.45, n)
            overlap = np.abs(r.normal(0.15, 0.35, n))
            stretch = r.normal(0.07, 0.03, n)
            cv = r.normal(0.15, 0.06, n)
            seam = r.normal(0.07, 0.03, n)
            cover = r.normal(0.72, 0.08, n)
    elif source == "artist":
        quad = r.uniform(0.82, 1.0, n)
        nonman = np.where(r.random(n) < 0.03, r.integers(1, 4, n), 0)
        mats = r.choice([1, 2, 3, 4], n, p=[0.5, 0.25, 0.15, 0.10])
        tex = r.choice([1024, 2048, 4096, 8192], n, p=[0.15, 0.45, 0.32, 0.08])
        islands_k = r.lognormal(np.log(2.5), 0.6, n)
        overlap = np.abs(r.normal(0.2, 0.5, n))
        stretch = r.normal(0.06, 0.03, n)
        cv = r.normal(0.14, 0.07, n)
        seam = r.normal(0.06, 0.03, n)
        cover = r.normal(0.74, 0.09, n)
    elif source == "objaverse":
        quad = r.uniform(0.15, 0.95, n)
        nonman = np.where(r.random(n) < 0.30, r.integers(1, 40, n), 0)
        mats = r.integers(1, 9, n)
        tex = r.choice([512, 1024, 2048, 4096, 8192], n, p=[0.1, 0.25, 0.3, 0.25, 0.1])
        islands_k = r.lognormal(np.log(9), 0.8, n)
        overlap = np.abs(r.normal(1.8, 2.5, n))
        stretch = r.normal(0.12, 0.07, n)
        cv = r.normal(0.28, 0.14, n)
        seam = r.normal(0.14, 0.08, n)
        cover = r.normal(0.6, 0.15, n)
    else:
        raise ValueError(source)

    tri = np.maximum(tri, 100)
    islands_k = _clip(islands_k, 0.05, 200)
    df = pd.DataFrame({
        "source": source,
        "tri_count": tri,
        "quad_ratio": _clip(quad, 0, 1),
        "non_manifold_edges": nonman,
        "islands_per_1k_tris": islands_k,
        "uv_island_count": np.maximum(1, np.round(islands_k * tri / 1000)).astype(int),
        "uv_overlap_pct": _clip(overlap, 0, 60),
        "uv_stretch": _clip(stretch, 0, 1),
        "texel_density_cv": _clip(cv, 0, 2),
        "seam_ratio": _clip(seam, 0, 1),
        "uv_coverage": _clip(cover, 0.05, 1),
        "material_slots": mats,
        "texture_res": tex,
        "platform": platform,
        "asset_role": role,
    })
    df["tri_budget_ratio"] = [
        t / TRI_BUDGET[p][a] for t, p, a in zip(df.tri_count, df.platform, df.asset_role)
    ]
    df["texture_budget_ratio"] = [t / TEXTURE_BUDGET[p] for t, p in zip(df.texture_res, df.platform)]
    return df


def rule_violations(row) -> list[str]:
    """SPEC features this row breaks."""
    out = []
    for feat, (kind, limit) in SPEC.items():
        v = row[feat]
        if (kind == "max" and v > limit) or (kind == "min" and v < limit):
            out.append(feat)
    return out


def _closest_margin(row) -> float:
    """Smallest relative distance to any threshold (0 = exactly on a threshold)."""
    margins = []
    for feat, (_, limit) in SPEC.items():
        scale = abs(limit) if limit else 1.0
        margins.append(abs(row[feat] - limit) / scale)
    return min(margins)


def label(df: pd.DataFrame, rng: np.random.Generator, noise_near=0.12, noise_far=0.01) -> pd.Series:
    """1 = FAIL (not production ready), 0 = PASS. Adds simulated TA disagreement."""
    rule_fail = df.apply(lambda r: len(rule_violations(r)) > 0, axis=1).astype(int)
    near = df.apply(_closest_margin, axis=1) < 0.10
    flip_p = np.where(near, noise_near, noise_far)
    flip = rng.random(len(df)) < flip_p
    return pd.Series(np.where(flip, 1 - rule_fail, rule_fail), index=df.index, name="label")


def make_dataset(n_per_source: int = 350, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    from .config import SOURCES

    df = pd.concat([_source_block(rng, s, n_per_source) for s in SOURCES], ignore_index=True)
    df["label"] = label(df, rng)
    df.insert(0, "asset_id", [f"asset_{i:05d}" for i in range(len(df))])
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)
