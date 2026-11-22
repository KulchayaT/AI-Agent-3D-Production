"""Report 4.1 interface: the same requests as postman/AI-Agent-3D-Production.postman_collection.json."""
import pytest
from fastapi.testclient import TestClient

from api.main import app
from qa_model.config import MODEL_PATH, ROOT
from qa_model.samples import make_samples

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="run scripts/train.py first")
client = TestClient(app)
URL = "/api/v1/assets/assess"
CTX = {"platform": "PC", "engine": "UE5", "asset_role": "hero_character", "target_fps": 60}


def body(file_url="samples/char_tripo_raw.glb", fmt="glb", **over):
    b = {"request_id": "3f2b9c1e-7a44-4d0b-9c61-0d5e8a7b2f10", "job_id": "job-0042",
         "asset_id": "char_cat_01", "timestamp": "2026-11-22T09:30:00Z", "attempt": 1,
         "mesh": {"file_url": file_url, "format": fmt}, "context": dict(CTX),
         "options": {"explain": True, "top_k_reasons": 3}}
    b.update(over)
    return b


@pytest.fixture(scope="module", autouse=True)
def samples():
    if not (ROOT / "samples" / "char_fixed.glb").exists():
        make_samples(ROOT / "samples")


def test_health():
    r = client.get("/api/v1/health")
    assert r.status_code == 200 and r.json()["model_ready"]


def test_t1_raw_tripo_mesh_fails_and_goes_back_to_modeler_c():
    r = client.post(URL, json=body())
    b = r.json()
    assert r.status_code == 200
    assert b["request_id"] == "3f2b9c1e-7a44-4d0b-9c61-0d5e8a7b2f10"
    assert b["verdict"] == "FAIL"
    assert any(i["code"] == "UV_FRAGMENTED" for i in b["issues"])
    assert b["next_action"]["agent"] == "3d_modeler_c"
    assert b["next_action"]["recommended_action"] == "automate"
    assert len(b["reasons"]) <= 3


def test_t1b_third_attempt_hands_off_to_human():
    r = client.post(URL, json=body(attempt=3))
    b = r.json()
    assert b["status"] == "needs_review"
    assert b["next_action"]["recommended_action"] == "prompt"


def test_t2_fixed_mesh_passes():
    r = client.post(URL, json=body("samples/char_fixed.glb"))
    b = r.json()
    assert r.status_code == 200
    assert b["verdict"] == "PASS" and b["readiness_score"] >= 80
    assert b["next_action"]["recommended_action"] == "annotate"


def test_t3_missing_platform_is_400():
    b = body()
    del b["context"]["platform"]
    r = client.post(URL, json=b)
    assert r.status_code == 400
    err = r.json()
    assert err["status"] == "error" and err["error"]["code"] == "MISSING_FIELD"
    assert err["error"]["field"] == "context.platform"
    assert err["request_id"] == b["request_id"]


def test_t3b_unknown_platform_is_422():
    b = body()
    b["context"]["platform"] = "Switch"
    r = client.post(URL, json=b)
    assert r.status_code == 422 and r.json()["error"]["code"] == "INVALID_VALUE"


def test_invalid_json_is_400():
    r = client.post(URL, content=b"{not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "INVALID_JSON"


def test_t4_image_file_is_415():
    r = client.post(URL, json=body("samples/concept.png", "glb"))
    assert r.status_code == 415 and r.json()["error"]["code"] == "UNSUPPORTED_FORMAT"


def test_t5_mesh_without_uv_is_422():
    r = client.post(URL, json=body("samples/no_uv.glb"))
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "MESH_HAS_NO_UV"


def test_missing_file_is_404():
    r = client.post(URL, json=body("samples/does_not_exist.glb"))
    assert r.status_code == 404 and r.json()["error"]["code"] == "FILE_NOT_FOUND"


def test_same_mesh_fails_on_mobile_prop_budget():
    b = body("samples/char_fixed.glb")
    b["context"] = {"platform": "Mobile", "engine": "UE5", "asset_role": "prop"}
    r = client.post(URL, json=b)
    assert r.json()["verdict"] == "FAIL"
    assert any(i["code"] == "OVER_TRI_BUDGET" for i in r.json()["issues"])


def test_feedback_is_201():
    r = client.post("/api/v1/feedback", json={
        "request_id": "3f2b9c1e-7a44-4d0b-9c61-0d5e8a7b2f10", "asset_id": "char_cat_01",
        "label": "wrong_pass", "source": "ta_review", "comment": "seam on the face",
        "timestamp": "2026-11-22T10:05:00Z"})
    assert r.status_code == 201 and r.json()["status"] == "recorded"


def test_job_flow():
    r = client.post("/api/v1/jobs", json={"prompt": "chibi cat adventurer", "context": CTX})
    assert r.status_code == 202
    job = client.get(r.json()["poll"]).json()
    assert job["stages"][0]["state"] in ("working", "done")
    assert client.get("/api/v1/jobs/unknown").status_code == 404
