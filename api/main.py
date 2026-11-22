"""3.6 + 4: REST API of the UV/Readiness QA model and a simulated production job.

Run:  uvicorn api.main:app --reload      Docs: http://127.0.0.1:8000/docs

Endpoints
  GET  /v1/health          model status
  POST /v1/assess          assess one asset (mesh file path or pre-computed features)
  POST /v1/jobs            start a production job (agent pipeline is SIMULATED in this PoC)
  GET  /v1/jobs/{job_id}   job progress for the Production Tracker screen
"""
from __future__ import annotations

import time
import uuid
from enum import Enum
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from qa_model.config import MESH_FEATURES, MODEL_PATH, ROOT
from qa_model.features import SUPPORTED_EXTENSIONS, MeshHasNoUV, load_mesh_features

app = FastAPI(title="AI Agent 3D Production Studio - QA API", version="0.1.0")


# ------------------------------------------------------------------ schemas
class Platform(str, Enum):
    PC = "PC"
    Console = "Console"
    Mobile = "Mobile"


class AssetRole(str, Enum):
    hero_character = "hero_character"
    npc_character = "npc_character"
    prop = "prop"


class Context(BaseModel):
    platform: Platform
    engine: str = "UE5"
    asset_role: AssetRole
    target_fps: int = Field(60, ge=15, le=240)


class AssessRequest(BaseModel):
    asset_id: str = Field(..., min_length=1, examples=["char_cat_01"])
    file_url: Optional[str] = Field(None, description="Local path to .glb/.gltf/.obj (relative to the project or absolute)")
    features: Optional[dict] = Field(None, description="Pre-computed mesh features (skips file reading)")
    context: Context


class JobRequest(BaseModel):
    prompt: str = Field(..., min_length=3)
    context: Context
    reference_image_url: Optional[str] = None


# ------------------------------------------------------------------ errors
def error(status: int, code: str, message: str, details=None) -> JSONResponse:
    return JSONResponse(status_code=status,
                        content={"error": {"code": code, "message": message, "details": details}})


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, exc: RequestValidationError):
    fields = [{"field": ".".join(str(p) for p in e["loc"][1:]), "problem": e["msg"]} for e in exc.errors()]
    return error(400, "INVALID_PAYLOAD", "request payload is missing or has invalid fields", fields)


@app.exception_handler(HTTPException)
async def http_error(_: Request, exc: HTTPException):
    detail = exc.detail if isinstance(exc.detail, dict) else {"code": "ERROR", "message": str(exc.detail)}
    return error(exc.status_code, detail.get("code", "ERROR"), detail.get("message", ""), detail.get("details"))


# ------------------------------------------------------------------ model
_model = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(503, {"code": "MODEL_NOT_READY",
                                      "message": "model file missing: run scripts/train.py"})
        from qa_model.predict import QAModel
        _model = QAModel(MODEL_PATH)
    return _model


def resolve_path(file_url: str) -> Path:
    raw = file_url.removeprefix("file://")
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


# ------------------------------------------------------------------ endpoints
@app.get("/v1/health")
def health():
    ready = MODEL_PATH.exists()
    out = {"status": "ok" if ready else "model_missing", "model_ready": ready}
    if ready:
        m = get_model()
        out.update({"model_version": m.version, "threshold": m.threshold})
    return out


@app.post("/v1/assess")
def assess(req: AssessRequest):
    if req.features is None and not req.file_url:
        raise HTTPException(400, {"code": "INVALID_PAYLOAD",
                                  "message": "send either file_url or features"})
    if req.features is not None:
        missing = [f for f in MESH_FEATURES if f not in req.features]
        if missing:
            raise HTTPException(400, {"code": "INVALID_PAYLOAD", "message": "features incomplete",
                                      "details": {"missing": missing}})
        features = req.features
    else:
        path = resolve_path(req.file_url)
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise HTTPException(415, {"code": "UNSUPPORTED_FORMAT",
                                      "message": f"'{path.suffix}' is not a mesh format we read",
                                      "details": {"supported": sorted(SUPPORTED_EXTENSIONS),
                                                  "hint": "convert FBX to GLB in the Blender worker first"}})
        if not path.exists():
            raise HTTPException(404, {"code": "FILE_NOT_FOUND", "message": str(req.file_url)})
        try:
            features = load_mesh_features(path)
        except MeshHasNoUV as e:
            raise HTTPException(422, {"code": "MESH_HAS_NO_UV", "message": str(e),
                                      "details": {"next_action": {"agent": "3d_modeler_c", "task": "unwrap_uv"}}})
        except Exception as e:  # unreadable file
            raise HTTPException(422, {"code": "MESH_UNREADABLE", "message": str(e)})

    model = get_model()
    t0 = time.perf_counter()
    result = model.assess(features, req.context.platform.value, req.context.asset_role.value)
    return {"asset_id": req.asset_id, **result,
            "latency_ms": round(1000 * (time.perf_counter() - t0), 1)}


# ------------------------------------------------------------------ simulated jobs
STAGES = [
    ("manager", "เขียน Asset Brief"),
    ("2d_art_director", "วาด Concept + Turnaround"),
    ("customer", "รอลูกค้า Approve Concept"),
    ("3d_modeler_a", "ตรวจภาพก่อนขึ้นโมเดล"),
    ("3d_modeler_b", "Image-to-3D"),
    ("3d_modeler_c", "จัด UV + Texture"),
    ("qa_model", "ตรวจ UV/Readiness"),
    ("technical_artist_a", "Import เข้า Unreal"),
    ("manager", "ตรวจรับตาม Brief"),
]
SECONDS_PER_STAGE = 3
JOBS: dict[str, dict] = {}


@app.post("/v1/jobs", status_code=202)
def create_job(req: JobRequest):
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {"created": time.time(), "prompt": req.prompt, "context": req.context.model_dump(mode="json")}
    return {"job_id": job_id, "status": "accepted", "poll": f"/v1/jobs/{job_id}",
            "note": "agent pipeline is simulated in this proof of concept"}


@app.get("/v1/jobs/{job_id}")
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, {"code": "JOB_NOT_FOUND", "message": job_id})
    done = int((time.time() - job["created"]) // SECONDS_PER_STAGE)
    stages = [{"agent": a, "task": t,
               "state": "done" if i < done else "working" if i == done else "waiting"}
              for i, (a, t) in enumerate(STAGES)]
    finished = done >= len(STAGES)
    return {"job_id": job_id, "status": "completed" if finished else "in_progress",
            "progress": min(1.0, round(done / len(STAGES), 2)), "stages": stages, **job}
