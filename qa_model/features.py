"""Measure UV and geometry features from a triangle mesh.

All measurements work on plain numpy arrays so they can be unit-tested without files:
    vertices (n, 3) float, faces (m, 3) int, uv (n, 2) float.
`load_mesh_features(path)` reads a .glb/.gltf/.obj file with trimesh and calls the same code.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import TEXTURE_BUDGET, TRI_BUDGET

SUPPORTED_EXTENSIONS = {".glb", ".gltf", ".obj"}


class MeshHasNoUV(ValueError):
    """The mesh has no UV coordinates, so UV quality cannot be measured."""


# ---------------------------------------------------------------- helpers
def _key_ids(points: np.ndarray, decimals: int = 5) -> np.ndarray:
    """Give every distinct (rounded) point one integer id."""
    rounded = np.round(points, decimals)
    _, ids = np.unique(rounded, axis=0, return_inverse=True)
    return ids.reshape(-1)


def _find(parent: np.ndarray, i: int) -> int:
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def _union(parent: np.ndarray, a: int, b: int) -> None:
    ra, rb = _find(parent, a), _find(parent, b)
    if ra != rb:
        parent[rb] = ra


def triangle_areas(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    p0, p1, p2 = (points[faces[:, i]] for i in range(3))
    if points.shape[1] == 2:
        e1, e2 = p1 - p0, p2 - p0
        return 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    return 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)


# ---------------------------------------------------------------- edges, islands, seams
def edge_stats(vertices: np.ndarray, faces: np.ndarray, uv: np.ndarray) -> dict:
    """Islands, seams and non-manifold edges.

    A 3D edge is identified by its two positions. Two faces belong to the same UV island
    when they share an edge whose positions AND UV coordinates both match. A 3D edge used
    by two faces whose UVs differ is a seam.
    """
    pos_id = _key_ids(vertices)
    uv_id = _key_ids(uv)
    m = len(faces)
    parent = np.arange(m)

    pos_edges: dict[tuple, list] = {}
    for f, tri in enumerate(faces):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            pa, pb = pos_id[a], pos_id[b]
            ua, ub = uv_id[a], uv_id[b]
            if pa > pb:
                pa, pb, ua, ub = pb, pa, ub, ua
            pos_edges.setdefault((pa, pb), []).append((f, ua, ub))

    seams = interior = non_manifold = 0
    for uses in pos_edges.values():
        if len(uses) > 2:
            non_manifold += 1
        if len(uses) == 2:
            interior += 1
            (f1, a1, b1), (f2, a2, b2) = uses
            if (a1, b1) == (a2, b2):
                _union(parent, f1, f2)
            else:
                seams += 1
        elif len(uses) > 2:
            # still join faces whose UVs match along the edge
            for i in range(len(uses)):
                for j in range(i + 1, len(uses)):
                    if uses[i][1:] == uses[j][1:]:
                        _union(parent, uses[i][0], uses[j][0])

    islands = len({_find(parent, i) for i in range(m)}) if m else 0
    return {
        "uv_island_count": islands,
        "seam_ratio": seams / interior if interior else 1.0,
        "non_manifold_edges": non_manifold,
    }


# ---------------------------------------------------------------- overlap and coverage
def uv_raster_stats(uv: np.ndarray, faces: np.ndarray, resolution: int = 256) -> dict:
    """Rasterise UV triangles on a grid over [0,1]^2.

    overlap_pct = share of covered pixels that are covered by more than one triangle.
    coverage    = share of the UV square that is covered at all.
    """
    count = np.zeros((resolution, resolution), dtype=np.int32)
    centers = (np.arange(resolution) + 0.5) / resolution
    tri_uv = uv[faces]  # (m, 3, 2)
    for t in tri_uv:
        lo = np.clip(np.floor(t.min(axis=0) * resolution).astype(int), 0, resolution - 1)
        hi = np.clip(np.ceil(t.max(axis=0) * resolution).astype(int), 0, resolution)
        if hi[0] <= lo[0] or hi[1] <= lo[1]:
            continue
        xs, ys = np.meshgrid(centers[lo[0]:hi[0]], centers[lo[1]:hi[1]], indexing="ij")
        (x0, y0), (x1, y1), (x2, y2) = t
        den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(den) < 1e-12:
            continue
        w0 = ((y1 - y2) * (xs - x2) + (x2 - x1) * (ys - y2)) / den
        w1 = ((y2 - y0) * (xs - x2) + (x0 - x2) * (ys - y2)) / den
        # strict test: a pixel centre lying exactly on an edge shared by two neighbours
        # must not count as overlap
        eps = 1e-9
        inside = (w0 > eps) & (w1 > eps) & (w0 + w1 < 1 - eps)
        count[lo[0]:hi[0], lo[1]:hi[1]] += inside
    covered = int((count >= 1).sum())
    overlapped = int((count >= 2).sum())
    return {
        "uv_overlap_pct": 100.0 * overlapped / covered if covered else 0.0,
        "uv_coverage": covered / resolution**2,
    }


# ---------------------------------------------------------------- stretch and texel density
def distortion_stats(vertices: np.ndarray, faces: np.ndarray, uv: np.ndarray) -> dict:
    """Per-face Jacobian from the 3D triangle to its UV triangle.

    uv_stretch       = area-weighted mean of (1 - smallest/largest singular value): 0 = no shear.
    texel_density_cv = area-weighted coefficient of variation of sqrt(uv_area / 3d_area).
    """
    p0, p1, p2 = (vertices[faces[:, i]] for i in range(3))
    q0, q1, q2 = (uv[faces[:, i]] for i in range(3))
    e1, e2 = p1 - p0, p2 - p0
    area3d = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
    ok = area3d > 1e-12
    if not ok.any():
        return {"uv_stretch": 0.0, "texel_density_cv": 0.0}
    e1, e2, area3d = e1[ok], e2[ok], area3d[ok]
    t1, t2 = (q1 - q0)[ok], (q2 - q0)[ok]

    len1 = np.linalg.norm(e1, axis=1)
    x_axis = e1 / len1[:, None]
    normal = np.cross(e1, e2)
    y_axis = np.cross(normal, e1)
    y_axis /= np.linalg.norm(y_axis, axis=1)[:, None]
    # local 2D coordinates of the 3D triangle: P rows = e1, e2
    P = np.zeros((len(e1), 2, 2))
    P[:, 0, 0] = len1
    P[:, 1, 0] = np.einsum("ij,ij->i", e2, x_axis)
    P[:, 1, 1] = np.einsum("ij,ij->i", e2, y_axis)
    T = np.stack([t1, t2], axis=1)  # (k, 2, 2)
    J = np.linalg.solve(P, T)
    s = np.linalg.svd(J, compute_uv=False)  # (k, 2), descending
    with np.errstate(divide="ignore", invalid="ignore"):
        shear = np.where(s[:, 0] > 1e-12, 1 - s[:, 1] / s[:, 0], 1.0)

    area_uv = triangle_areas(uv, faces)[ok]
    density = np.sqrt(area_uv / area3d)
    w = area3d / area3d.sum()
    mean = float((w * density).sum())
    std = float(np.sqrt((w * (density - mean) ** 2).sum()))
    return {
        "uv_stretch": float((w * shear).sum()),
        "texel_density_cv": std / mean if mean > 0 else 1.0,
    }


# ---------------------------------------------------------------- quads
def quad_ratio(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Estimate how much of a triangulated mesh came from quads.

    Two triangles count as one quad when they are coplanar and their shared edge is the
    longest edge of both (the diagonal). Greedy pairing, each face used once.
    """
    m = len(faces)
    if m == 0:
        return 0.0
    pos_id = _key_ids(vertices)
    f_pos = pos_id[faces]
    normals = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                       vertices[faces[:, 2]] - vertices[faces[:, 0]])
    norms = np.linalg.norm(normals, axis=1)
    normals = normals / np.where(norms > 0, norms, 1)[:, None]
    edge_len = np.stack([
        np.linalg.norm(vertices[faces[:, (i + 1) % 3]] - vertices[faces[:, i]], axis=1)
        for i in range(3)
    ], axis=1)
    longest = edge_len.argmax(axis=1)

    diag: dict[tuple, list] = {}
    for f in range(m):
        i = longest[f]
        a, b = f_pos[f, i], f_pos[f, (i + 1) % 3]
        diag.setdefault((min(a, b), max(a, b)), []).append(f)
    used = np.zeros(m, dtype=bool)
    pairs = 0
    for fs in diag.values():
        if len(fs) == 2:
            f1, f2 = fs
            if not used[f1] and not used[f2] and abs(np.dot(normals[f1], normals[f2])) > 0.999:
                used[f1] = used[f2] = True
                pairs += 1
    return 2 * pairs / m


# ---------------------------------------------------------------- public API
def compute_mesh_features(vertices, faces, uv, material_slots: int = 1,
                          texture_res: int = 1024) -> dict:
    """All MESH_FEATURES for one mesh."""
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=int)
    if uv is None or len(uv) != len(vertices):
        raise MeshHasNoUV("mesh has no UV coordinates")
    uv = np.asarray(uv, dtype=float)
    m = len(faces)
    feats = {"tri_count": m, "quad_ratio": quad_ratio(vertices, faces)}
    feats.update(edge_stats(vertices, faces, uv))
    feats["islands_per_1k_tris"] = 1000.0 * feats["uv_island_count"] / m if m else 0.0
    feats.update(uv_raster_stats(uv, faces))
    feats.update(distortion_stats(vertices, faces, uv))
    feats["material_slots"] = int(material_slots)
    feats["texture_res"] = int(texture_res)
    return feats


def add_context_features(features: dict, platform: str, asset_role: str) -> dict:
    """Add budget ratios that depend on the target platform and asset role."""
    out = dict(features)
    out["platform"] = platform
    out["asset_role"] = asset_role
    out["tri_budget_ratio"] = features["tri_count"] / TRI_BUDGET[platform][asset_role]
    out["texture_budget_ratio"] = features["texture_res"] / TEXTURE_BUDGET[platform]
    return out


def load_mesh_features(path: str | Path) -> dict:
    """Read a .glb/.gltf/.obj file and measure it. Raises MeshHasNoUV if it has no UVs."""
    import trimesh  # imported here so the pure functions work without trimesh

    path = Path(path)
    scene = trimesh.load(path, force="scene", process=False)
    meshes = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
    if not meshes:
        raise ValueError("file contains no triangle mesh")

    verts, faces, uvs, offset = [], [], [], 0
    texture_res = 0
    for g in meshes:
        uv = getattr(g.visual, "uv", None)
        if uv is None or len(uv) != len(g.vertices):
            raise MeshHasNoUV(f"mesh '{g.metadata.get('name', '?')}' has no UV coordinates")
        verts.append(np.asarray(g.vertices))
        faces.append(np.asarray(g.faces) + offset)
        uvs.append(np.asarray(uv))
        offset += len(g.vertices)
        material = getattr(g.visual, "material", None)
        for attr in ("image", "baseColorTexture"):
            img = getattr(material, attr, None)
            # ignore tiny placeholder images (trimesh writes one when a mesh has UVs but no texture)
            if img is not None and hasattr(img, "size") and max(img.size) >= 64:
                texture_res = max(texture_res, *img.size)
    return compute_mesh_features(
        np.vstack(verts), np.vstack(faces), np.vstack(uvs),
        material_slots=len(meshes),
        texture_res=texture_res or 1024,  # no texture embedded: assume 1K
    )
