"""Small test meshes that imitate the cases in the report.

char_fixed     open cylinder with one clean unwrap: 1 island, one seam, even texel density
char_tripo_raw the same cylinder where every triangle is its own scattered, overlapping island
no_uv          the same cylinder with no UV coordinates at all
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def cylinder(segments: int = 48, rings: int = 24, radius: float = 0.3, height: float = 1.8):
    """Open cylinder with a UV seam: the first column of vertices is duplicated at u = 1."""
    verts, uv = [], []
    for j in range(rings + 1):
        v = j / rings
        for i in range(segments + 1):
            u = i / segments
            a = 2 * np.pi * u
            verts.append((radius * np.cos(a), radius * np.sin(a), height * v))
            uv.append((u, v))
    faces = []
    row = segments + 1
    for j in range(rings):
        for i in range(segments):
            a, b = j * row + i, j * row + i + 1
            c, d = a + row, b + row
            faces += [(a, b, d), (a, d, c)]
    verts, uv, faces = np.array(verts), np.array(uv), np.array(faces)
    # keep texel density even: circumference 2*pi*r maps to u in [0,1], height to v in [0,1]
    circ = 2 * np.pi * radius
    scale = 1.0 / max(circ, height)
    uv = np.column_stack([uv[:, 0] * circ * scale, uv[:, 1] * height * scale])
    return verts, faces, uv


def scatter_uv(vertices, faces, seed: int = 0):
    """Every triangle becomes its own island at a random place, size and rotation."""
    rng = np.random.default_rng(seed)
    new_v, new_uv, new_f = [], [], []
    for k, tri in enumerate(faces):
        center = rng.uniform(0.1, 0.9, 2)
        size = rng.uniform(0.02, 0.12)
        ang = rng.uniform(0, 2 * np.pi)
        base = np.array([[0, 0], [1, 0], [rng.uniform(0, 1), rng.uniform(0.3, 1)]]) - 0.4
        rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        pts = center + size * base @ rot.T
        for idx, p in zip(tri, pts):
            new_v.append(vertices[idx])
            new_uv.append(p)
        new_f.append((3 * k, 3 * k + 1, 3 * k + 2))
    return np.array(new_v), np.array(new_f), np.clip(np.array(new_uv), 0, 1)


def write_glb(path: Path, vertices, faces, uv=None) -> Path:
    import trimesh

    if uv is None:
        mesh = trimesh.Trimesh(vertices, faces, process=False)
    else:
        mesh = trimesh.Trimesh(vertices, faces, process=False,
                               visual=trimesh.visual.TextureVisuals(uv=uv))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)
    return path


def make_samples(out_dir: Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    v, f, uv = cylinder()
    rv, rf, ruv = scatter_uv(v, f)
    return {
        "char_fixed": write_glb(out_dir / "char_fixed.glb", v, f, uv),
        "char_tripo_raw": write_glb(out_dir / "char_tripo_raw.glb", rv, rf, ruv),
        "no_uv": write_glb(out_dir / "no_uv.glb", v, f, None),
    }
