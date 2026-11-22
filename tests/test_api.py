"""4.2 scenarios T1-T5 (the same requests as postman/collection.json)."""
import pytest
from fastapi.testclient import TestClient

from api.main import app
from qa_model.config import MODEL_PATH, ROOT
from qa_model.samples import make_samples

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="run scripts/train.py first")
client = TestClient(app)
CTX = {"platform": "PC", "engine": "UE5", "asset_role": "hero_character"}


@pytest.fixture(scope="module", autouse=True)
def samples():
    if not (ROOT / "samples" / "char_fixed.glb").exists():
        make_samples(ROOT / "samples")


def test_health():
    r = client.get("/v1/health")
    assert r.status_code == 200 and r.json()["model_ready"]


def test_t1_raw_tripo_mesh_fails_with_uv_reason():
    r = client.post("/v1/assess", json={"asset_id": "t1", "file_url": "samples/char_tripo_raw.glb", "context": CTX})
    body = r.json()
    assert r.status_code == 200
    assert body["status"] == "FAIL"
    assert any(i["code"] == "UV_FRAGMENTED" for i in body["issues"])
    assert body["next_action"]["agent"] == "3d_modeler_c"
    assert body["reasons"]


def test_t2_fixed_mesh_passes():
    r = client.post("/v1/assess", json={"asset_id": "t2", "file_url": "samples/char_fixed.glb", "context": CTX})
    body = r.json()
    assert r.status_code == 200
    assert body["status"] == "PASS"
    assert body["readiness_score"] >= 80


def test_t3_missing_platform_is_400():
    ctx = {k: v for k, v in CTX.items() if k != "platform"}
    r = client.post("/v1/assess", json={"asset_id": "t3", "file_url": "samples/char_fixed.glb", "context": ctx})
    assert r.status_code == 400
    assert any(d["field"] == "context.platform" for d in r.json()["error"]["details"])


def test_t4_image_file_is_415():
    r = client.post("/v1/assess", json={"asset_id": "t4", "file_url": "samples/concept.png", "context": CTX})
    assert r.status_code == 415


def test_t5_mesh_without_uv_is_422():
    r = client.post("/v1/assess", json={"asset_id": "t5", "file_url": "samples/no_uv.glb", "context": CTX})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "MESH_HAS_NO_UV"


def test_same_mesh_fails_on_mobile_prop_budget():
    ctx = {"platform": "Mobile", "engine": "UE5", "asset_role": "prop"}
    r = client.post("/v1/assess", json={"asset_id": "t6", "file_url": "samples/char_fixed.glb", "context": ctx})
    body = r.json()
    assert body["status"] == "FAIL"
    assert any(i["code"] == "OVER_TRI_BUDGET" for i in body["issues"])


def test_job_flow():
    r = client.post("/v1/jobs", json={"prompt": "chibi cat adventurer", "context": CTX})
    assert r.status_code == 202
    job = client.get(r.json()["poll"]).json()
    assert job["stages"][0]["state"] in ("working", "done")
    assert client.get("/v1/jobs/unknown").status_code == 404
