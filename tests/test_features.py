"""Feature extractor checked on meshes whose answer we know."""
import numpy as np
import pytest

from qa_model.features import (MeshHasNoUV, compute_mesh_features, distortion_stats,
                               edge_stats, quad_ratio, uv_raster_stats)
from qa_model.samples import cylinder, scatter_uv

# unit square made of 2 triangles
SQ_V = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
SQ_F = np.array([[0, 1, 2], [0, 2, 3]])
SQ_UV = SQ_V[:, :2] * 0.5


def test_clean_square_has_one_island_no_overlap_no_stretch():
    e = edge_stats(SQ_V, SQ_F, SQ_UV)
    assert e["uv_island_count"] == 1
    assert e["seam_ratio"] == 0
    r = uv_raster_stats(SQ_UV, SQ_F)
    assert r["uv_overlap_pct"] == pytest.approx(0, abs=0.5)
    assert r["uv_coverage"] == pytest.approx(0.25, abs=0.02)
    d = distortion_stats(SQ_V, SQ_F, SQ_UV)
    assert d["uv_stretch"] == pytest.approx(0, abs=1e-6)
    assert d["texel_density_cv"] == pytest.approx(0, abs=1e-6)


def test_stacked_uv_is_detected_as_overlap():
    # two separate quads whose UVs land on exactly the same place
    v = np.vstack([SQ_V, SQ_V + [2, 0, 0]])
    f = np.vstack([SQ_F, SQ_F + 4])
    uv = np.vstack([SQ_UV, SQ_UV])
    r = uv_raster_stats(uv, f)
    assert r["uv_overlap_pct"] > 95


def test_stretched_uv_has_shear():
    uv = SQ_V[:, :2] * [1.0, 0.25]  # squash v
    d = distortion_stats(SQ_V, SQ_F, uv)
    assert d["uv_stretch"] == pytest.approx(0.75, abs=1e-6)


def test_square_is_one_quad():
    assert quad_ratio(SQ_V, SQ_F) == 1.0


def test_cylinder_vs_scattered_islands():
    v, f, uv = cylinder(segments=16, rings=8)
    good = compute_mesh_features(v, f, uv)
    assert good["uv_island_count"] == 1
    assert good["uv_overlap_pct"] < 1
    assert good["seam_ratio"] < 0.1
    rv, rf, ruv = scatter_uv(v, f)
    bad = compute_mesh_features(rv, rf, ruv)
    assert bad["uv_island_count"] == len(f)  # every triangle is its own island
    assert bad["seam_ratio"] == 1.0
    assert bad["uv_overlap_pct"] > good["uv_overlap_pct"]


def test_non_manifold_edge_counted():
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1]], float)
    f = np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]])  # edge 0-1 used by 3 faces
    uv = v[:, :2]
    assert edge_stats(v, f, uv)["non_manifold_edges"] == 1


def test_missing_uv_raises():
    with pytest.raises(MeshHasNoUV):
        compute_mesh_features(SQ_V, SQ_F, None)
