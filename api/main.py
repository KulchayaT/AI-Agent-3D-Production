"""Section 4.1 UI-Model interface: REST API of the UV/Readiness QA model (AI Task 1)
and the production-job endpoints the Web Portal polls (AI Task 2, simulated here).

Run:  uvicorn api.main:app --reload      Docs: http://127.0.0.1:8000/docs

Endpoints (report 4.1)
  POST /api/v1/assets/assess   assess one 3D asset (mesh file or pre-computed features)
  POST /api/v1/feedback        record a TA / customer verdict on an assessment   -> 201
  GET  /api/v1/health          model version and readiness
  POST /api/v1/jobs            start a production job (agent team SIMULATED)    -> 202
  GET  /api/v1/jobs/{job_id}   job progress for the Production Tracker screen

Every error uses one body shape:
  {"request_id": "...", "status": "error", "error": {"code": "...", "message": "...", "field": "..."}}
"""
from __future__ import annotations

import time
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from qa_model.config import MESH_FEATURES, MODEL_PATH, ROOT
from qa_model.features import SUPPORTED_EXTENSIONS, MeshHasNoUV, load_mesh_features

API = "/api/v1"
MAX_FILE_MB = 50          # 413 above this
MAX_ATTEMPTS = 3          # SPEC-4: Modeler C may re-unwrap at most 3 times, then a human TA
REVIEW_BAND = (0.35, 0.65)  # fail probability where the model is unsure -> human review

app = FastAPI(title="AI Agent 3D Production Studio - QA API", version="1.0.0")


# ------------------------------------------------------------------ request schemas
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
    engine: Literal["UE5"] = "UE5"
    asset_role: AssetRole
    target_fps: int = Field(60, ge=15, le=240)


class Mesh(BaseModel):
    file_url: str = Field(..., min_length=1, description="Path to .glb/.gltf/.obj (relative to the project or absolute)")
    format: Literal["glb", "gltf", "obj"]


class Options(BaseModel):
    explain: bool = True
    top_k_reasons: int = Field(3, ge=1, le=10)


class AssessRequest(BaseModel):
    request_id: str = Field(..., min_length=8, max_length=64)
    job_id: str = Field(..., min_length=1, max_length=32)
    asset_id: str = Field(..., min_length=1, max_length=64)
    timestamp: datetime
    attempt: int = Field(1, ge=1, le=MAX_ATTEMPTS)
    mesh: Optional[Mesh] = None
    features: Optional[dict] = Field(None, description="Pre-computed features from the Blender worker")
    context: Context
    options: Options = Options()


class FeedbackRequest(BaseModel):
    request_id: str = Field(..., min_length=8, max_length=64)
    asset_id: str = Field(..., min_length=1, max_length=64)
    label: Literal["agree", "wrong_pass", "wrong_fail"]
    source: Literal["ta_review", "customer_portal", "manager_agent"]
    comment: Optional[str] = Field(None, max_length=500)
    timestamp: datetime


class JobRequest(BaseModel):
    prompt: str = Field(..., min_length=3, max_length=2000)
    context: Context
    reference_image_url: Optional[str] = None


# ------------------------------------------------------------------ errors
class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, field: str | None = None, details=None):
        self.status, self.code, self.message, self.field, self.details = status, code, message, field, details


def error_body(request_id, code, message, field=None, details=None) -> dict:
    err = {"code": code, "message": message, "field": field}
    if details is not None:
        err["details"] = details
    return {"request_id": request_id, "status": "error", "error": err}


async def _request_id(request: Request):
    try:
        body = await request.json()
        return body.get("request_id") if isinstance(body, dict) else None
    except Exception:
        return None


@app.exception_handler(ApiError)
async def api_error(request: Request, exc: ApiError):
    return JSONResponse(status_code=exc.status,
                        content=error_body(await _request_id(request), exc.code, exc.message, exc.field, exc.details))


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    """400 = body is not JSON or a required field is missing; 422 = a value is out of range."""
    errors = exc.errors()
    first = errors[0]
    field = ".".join(str(p) for p in first["loc"][1:]) or None
    if first["type"] == "json_invalid":
        status, code, msg = 400, "INVALID_JSON", "request body is not valid JSON"
    elif any(e["type"] == "missing" for e in errors):
        miss = next(e for e in errors if e["type"] == "missing")
        field = ".".join(str(p) for p in miss["loc"][1:])
        status, code, msg = 400, "MISSING_FIELD", f"{field} is required"
    else:
        status, code, msg = 422, "INVALID_VALUE", f"{field}: {first['msg']}"
    return JSONResponse(status_code=status, content=error_body(await _request_id(request), code, msg, field))


@app.exception_handler(HTTPException)
async def http_error(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code,
                        content=error_body(await _request_id(request), "HTTP_ERROR", str(exc.detail)))


# ------------------------------------------------------------------ model
_model = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise ApiError(503, "MODEL_NOT_READY", "model file missing: run scripts/train.py")
        from qa_model.predict import QAModel
        _model = QAModel(MODEL_PATH)
    return _model


def resolve_path(file_url: str) -> Path:
    path = Path(file_url.removeprefix("file://"))
    return path if path.is_absolute() else ROOT / path


def recommended_action(result: dict, attempt: int) -> str:
    """Presenting intelligence (report 2.3.1): annotate / automate / prompt."""
    if result["status"] == "PASS" and not result["needs_human_review"]:
        return "annotate"                     # show the report, pipeline continues
    if result["status"] == "FAIL" and result["issues"] and attempt < MAX_ATTEMPTS \
            and not result["needs_human_review"]:
        return "automate"                     # send back to the agent that fixes it
    return "prompt"                           # ask a human Technical Artist


# ------------------------------------------------------------------ endpoints
@app.get(f"{API}/health")
def health():
    ready = MODEL_PATH.exists()
    out = {"status": "ok" if ready else "unavailable", "model_ready": ready}
    if ready:
        m = get_model()
        out.update({"model_version": m.version, "threshold": m.threshold})
    return out


@app.post(f"{API}/assets/assess")
def assess(req: AssessRequest):
    t0 = time.perf_counter()
    if req.features is None and req.mesh is None:
        raise ApiError(400, "MISSING_FIELD", "send either mesh or features", "mesh")
    if req.features is not None:
        missing = [f for f in MESH_FEATURES if f not in req.features]
        if missing:
            raise ApiError(400, "MISSING_FIELD", f"features.{missing[0]} is required",
                           f"features.{missing[0]}", {"missing": missing})
        features = req.features
    else:
        path = resolve_path(req.mesh.file_url)
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS or path.suffix.lower().lstrip(".") != req.mesh.format:
            raise ApiError(415, "UNSUPPORTED_FORMAT",
                           f"'{path.suffix}' is not a mesh format we read or does not match mesh.format",
                           "mesh.file_url", {"supported": sorted(SUPPORTED_EXTENSIONS),
                                             "hint": "convert FBX to GLB in the Blender worker first"})
        if not path.exists():
            raise ApiError(404, "FILE_NOT_FOUND", f"{req.mesh.file_url} does not exist", "mesh.file_url")
        size_mb = path.stat().st_size / 1e6
        if size_mb > MAX_FILE_MB:
            raise ApiError(413, "FILE_TOO_LARGE", f"{size_mb:.1f} MB > {MAX_FILE_MB} MB", "mesh.file_url")
        try:
            features = load_mesh_features(path)
        except MeshHasNoUV as e:
            raise ApiError(422, "MESH_HAS_NO_UV", str(e), "mesh.file_url",
                           {"next_action": {"agent": "3d_modeler_c", "task": "unwrap_uv"}})
        except Exception as e:  # file cannot be parsed
            raise ApiError(422, "MESH_UNREADABLE", str(e), "mesh.file_url")
    t1 = time.perf_counter()

    model = get_model()
    try:
        result = model.assess(features, req.context.platform.value, req.context.asset_role.value,
                              explain=req.options.explain, top_k=req.options.top_k_reasons)
    except Exception as e:
        raise ApiError(500, "INFERENCE_ERROR", f"model failed: {e}")
    t2 = time.perf_counter()

    action = recommended_action(result, req.attempt)
    next_action = result["next_action"]
    if action == "prompt":
        next_action = {"agent": "technical_artist_human", "task": "manual_review"}
    return {
        "request_id": req.request_id,
        "job_id": req.job_id,
        "asset_id": req.asset_id,
        "status": "needs_review" if action == "prompt" else "ok",
        "model_version": result["model_version"],
        "verdict": result["status"],
        "readiness_score": result["readiness_score"],
        "fail_probability": result["fail_probability"],
        "threshold": model.threshold,
        "issues": result["issues"],
        "reasons": result["reasons"],
        "next_action": {**next_action, "recommended_action": action,
                        "attempt": req.attempt, "max_attempts": MAX_ATTEMPTS},
        "features": result["features"],
        "latency_ms": {"feature_extraction": round(1000 * (t1 - t0), 1),
                       "inference_and_explain": round(1000 * (t2 - t1), 1),
                       "total": round(1000 * (t2 - t0), 1)},
    }


FEEDBACK: list[dict] = []


@app.post(f"{API}/feedback", status_code=201)
def feedback(req: FeedbackRequest):
    """Human verdicts become new training labels (report 2.3.1 User feedback)."""
    fb_id = f"fb-{len(FEEDBACK) + 1:06d}"
    FEEDBACK.append({"feedback_id": fb_id, **req.model_dump(mode="json")})
    return {"feedback_id": fb_id, "status": "recorded"}


# ------------------------------------------------------------------ simulated jobs (AI Task 2)
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


@app.post(f"{API}/jobs", status_code=202)
def create_job(req: JobRequest):
    job_id = "job-" + uuid.uuid4().hex[:8]
    JOBS[job_id] = {"created": time.time(), "prompt": req.prompt, "context": req.context.model_dump(mode="json")}
    return {"job_id": job_id, "status": "accepted", "poll": f"{API}/jobs/{job_id}",
            "note": "agent pipeline is simulated in this proof of concept"}


@app.get(f"{API}/jobs/{{job_id}}")
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise ApiError(404, "JOB_NOT_FOUND", f"{job_id} does not exist", "job_id")
    done = int((time.time() - job["created"]) // SECONDS_PER_STAGE)
    stages = [{"agent": a, "task": t,
               "state": "done" if i < done else "working" if i == done else "waiting"}
              for i, (a, t) in enumerate(STAGES)]
    finished = done >= len(STAGES)
    return {"job_id": job_id, "status": "completed" if finished else "in_progress",
            "progress": min(1.0, round(done / len(STAGES), 2)), "stages": stages, **job}
