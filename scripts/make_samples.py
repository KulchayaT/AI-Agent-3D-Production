"""Write the sample meshes in samples/ and print their measured features."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qa_model.config import ROOT  # noqa: E402
from qa_model.features import MeshHasNoUV, load_mesh_features  # noqa: E402
from qa_model.samples import make_samples  # noqa: E402

if __name__ == "__main__":
    for name, path in make_samples(ROOT / "samples").items():
        try:
            feats = load_mesh_features(path)
            print(name, json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in feats.items()}))
        except MeshHasNoUV as e:
            print(name, "->", e)
