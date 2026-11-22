# AI Agent 3D Production Studio — UV/Readiness QA Model

BD532 Preliminary Report (AI Independent Study) · Kulchaya Sinsiriwanich 68130014

ทีม AI Agent 8 ตำแหน่งที่สร้างตัวละคร 3D จนใช้งานได้ใน Unreal Engine 5
Repository นี้คือโค้ดประกอบรายงาน: **AI Task 1 — UV/Readiness QA** (โมเดล ML ที่ตัดสินว่าโมเดล 3D
"พร้อมเข้า Engine" หรือไม่ พร้อมเหตุผล SHAP และ Agent ที่ต้องแก้) และ API ตามที่ออกแบบใน **Section 4.1**
ทีม Agent (AI Task 2) ถูกจำลองด้วย Job endpoint

```
Customer -> Manager -> 2D Art Director -> Customer approves concept -> 3D Modeler A (check views)
         -> 3D Modeler B (Image-to-3D) -> 3D Modeler C (UV + Texture) -> [AI Task 1: UV/Readiness QA]  <- this repo
         -> Technical Artist A (Unreal import) -> Manager review -> Customer download + feedback
```

| Report section | Code |
| --- | --- |
| 3.1 Basic input data exploration | `scripts/eda.py` → `reports/eda_*.png` |
| 4.1 UI-Model interface design | `api/main.py`, `postman/`, `tests/test_api.py` |
| Supporting (model behind the API) | `scripts/train.py` (4 versions in MLflow), `scripts/explain_model.py` (SHAP, fairness) |

> **Proof of concept:** training data in `data/assets.csv` is **synthetic** (`qa_model/data.py`),
> generated from the SPEC thresholds with simulated Technical-Artist disagreement. The feature
> extractor is real and works on .glb/.gltf/.obj files; replace the synthetic rows with features
> extracted from real Tripo outputs + TA labels in the IS phase. The agent pipeline behind
> `/api/v1/jobs` is simulated.

## Project layout

| Path | What it is |
| --- | --- |
| `qa_model/features.py` | Measures a mesh: UV islands, overlap %, stretch, texel-density CV, seams, coverage, quad ratio, non-manifold edges |
| `qa_model/data.py` | Synthetic dataset (4 sources: `tripo_raw`, `tripo_fixed`, `artist`, `objaverse`) + SPEC labelling |
| `qa_model/model.py` | The 4 model versions compared in MLflow |
| `qa_model/explain.py`, `predict.py` | SHAP reasons in Thai, issues, next agent to call |
| `qa_model/config.py` | SPEC thresholds, triangle/texture budgets per platform and asset role |
| `scripts/` | `generate_data`, `eda`, `train`, `explain_model`, `make_samples` |
| `api/main.py` | FastAPI service (`/api/v1/assets/assess`, `/api/v1/feedback`, `/api/v1/health`, `/api/v1/jobs`) |
| `postman/` | Postman collection for the 4.1 scenarios |
| `tests/` | pytest: feature extractor, data, API (23 tests) |
| `samples/` | Test meshes: `char_fixed.glb`, `char_tripo_raw.glb`, `no_uv.glb` |
| `reports/` | EDA plots, model comparison, confusion matrices, SHAP plots, fairness |
| `models/qa_model.joblib` | Selected model |
| `mlruns.db`, `mlruns/` | MLflow tracking store |

## Setup (Windows PowerShell)

Python 3.10+.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS / Linux: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`

## Run everything

```powershell
python scripts/generate_data.py     # data/assets.csv (1,400 rows)
python scripts/eda.py               # 3.1 data types, statistics, class distribution -> reports/eda_*.png
python scripts/train.py             # train 4 versions, log to MLflow, save best model
python scripts/explain_model.py     # fairness, SHAP summary, explained cases
python scripts/make_samples.py      # sample meshes for the API tests
pytest -q                           # unit + API tests (23 tests)
```

MLflow UI: `mlflow ui --backend-store-uri sqlite:///mlruns.db`, then open http://127.0.0.1:5000.

## 3.1 Data exploration (results of `scripts/eda.py`)

1,400 assets, 16 features + label, no missing values, stratified split train 1,120 / test 280.

| Source | Assets | FAIL | FAIL rate |
| --- | --- | --- | --- |
| tripo_raw | 350 | 304 | 86.9% |
| tripo_fixed | 350 | 150 | 42.9% |
| artist | 350 | 177 | 50.6% |
| objaverse | 350 | 310 | 88.6% |
| total | 1,400 | 941 | 67.2% |

FAIL rate by platform: Mobile 84.5%, Console 69.0%, PC 58.3%. Most broken rules: texel density (39% of
assets), UV overlap (37%), triangle budget (34%), fragmented islands (34%). Because predicting FAIL for
everything already gives 67% accuracy, the model is judged by Recall(FAIL) per source, not accuracy.

## Model behind the API (synthetic data, test split 20% = 280 assets)

| Version | Features | Threshold | Test Recall (FAIL) | Test Precision | Test F1 | ROC AUC |
| --- | --- | --- | --- | --- | --- | --- |
| v1 Logistic Regression | mesh only | 0.23 | 0.989 | 0.684 | 0.809 | 0.749 |
| v2 Random Forest | mesh only | 0.23 | 0.888 | 0.746 | 0.811 | 0.763 |
| v3 XGBoost | mesh only | 0.29 | 0.957 | 0.729 | 0.828 | 0.766 |
| **v4 XGBoost + context** | mesh + platform/role budgets | 0.24 | **0.936** | **0.834** | **0.882** | **0.808** |

Selected: v4 (best cross-validated F1 among versions with cross-validated Recall(FAIL) ≥ 0.90).
Not met yet: Cohen's kappa 0.59 (goal 0.70), to be re-measured with real TA labels.

## 4.1 API

```powershell
uvicorn api.main:app --reload
```

Swagger docs: http://127.0.0.1:8000/docs

| Method | Endpoint | Purpose |
| --- | --- | --- |
| POST | `/api/v1/assets/assess` | Assess one 3D asset (mesh file or pre-computed features) + context |
| POST | `/api/v1/feedback` | Record a TA / customer verdict → 201 |
| GET | `/api/v1/health` | Model version and readiness |
| POST | `/api/v1/jobs` | Start a production job (simulated agents) → 202 |
| GET | `/api/v1/jobs/{job_id}` | Job progress for the Production Tracker |

### Request

```json
{
  "request_id": "3f2b9c1e-7a44-4d0b-9c61-0d5e8a7b2f10",
  "job_id": "job-0042",
  "asset_id": "char_cat_01",
  "timestamp": "2026-11-22T09:30:00Z",
  "attempt": 1,
  "mesh": { "file_url": "samples/char_tripo_raw.glb", "format": "glb" },
  "context": { "platform": "PC", "engine": "UE5", "asset_role": "hero_character", "target_fps": 60 },
  "options": { "explain": true, "top_k_reasons": 3 }
}
```

Send `mesh` **or** `features` (the 12 mesh features already extracted by the Blender worker).
`attempt` is 1–3 (SPEC-6 loop limit). `platform`: `PC` | `Console` | `Mobile`;
`asset_role`: `hero_character` | `npc_character` | `prop`.

### Response 200 (shortened)

```json
{
  "request_id": "3f2b9c1e-7a44-4d0b-9c61-0d5e8a7b2f10",
  "status": "ok",
  "model_version": "v4_xgboost_context",
  "verdict": "FAIL",
  "readiness_score": 13,
  "fail_probability": 0.8731,
  "threshold": 0.24,
  "issues": [{ "code": "UV_FRAGMENTED", "value": 1000.0, "limit": 8.0, "agent": "3d_modeler_c", "task": "re_unwrap" }],
  "reasons": [{ "feature": "seam_ratio", "value": 1.0, "shap": 0.6983, "text": "มี Seam มากเกินไป" }],
  "next_action": { "agent": "3d_modeler_c", "task": "re_unwrap", "recommended_action": "automate", "attempt": 1, "max_attempts": 3 },
  "latency_ms": { "feature_extraction": 876.1, "inference_and_explain": 2008.6, "total": 2884.7 }
}
```

`recommended_action` follows the report's presenting-intelligence rule: `annotate` (PASS, continue),
`automate` (FAIL with a clear issue, send back to the agent, attempt < 3), `prompt` (unsure 0.35–0.65,
or the third attempt → `status: needs_review`, human TA). The first request takes longer because the
SHAP explainer is built.

### Errors

One body shape: `{"request_id": "...", "status": "error", "error": {"code": "...", "message": "...", "field": "..."}}`

| Status | Code | When |
| --- | --- | --- |
| 400 | `INVALID_JSON`, `MISSING_FIELD` | body is not JSON, a required field is missing, or neither `mesh` nor `features` |
| 404 | `FILE_NOT_FOUND`, `JOB_NOT_FOUND` | file or job does not exist |
| 413 | `FILE_TOO_LARGE` | mesh file over 50 MB (send `features` instead) |
| 415 | `UNSUPPORTED_FORMAT` | not .glb/.gltf/.obj or does not match `mesh.format` |
| 422 | `INVALID_VALUE`, `MESH_HAS_NO_UV`, `MESH_UNREADABLE` | value out of range / unknown enum, or the mesh cannot be assessed |
| 500 | `INFERENCE_ERROR` | the model raised an exception |
| 503 | `MODEL_NOT_READY` | `models/qa_model.joblib` missing |

### Postman

Import `postman/AI-Agent-3D-Production.postman_collection.json`, start the API, run the collection.
The same scenarios run in `tests/test_api.py`:

| # | Scenario | Expected |
| --- | --- | --- |
| T1 | raw Tripo-like mesh | 200 FAIL, `UV_FRAGMENTED`, `automate` → `3d_modeler_c` |
| T1b | same mesh, attempt 3 | `needs_review`, `prompt` |
| T2 | same mesh after a clean unwrap | 200 PASS, score ≥ 80, `annotate` |
| T3 | `context.platform` missing | 400 `MISSING_FIELD`, field `context.platform` |
| T3b | `context.platform` = `Switch` | 422 `INVALID_VALUE` |
| T4 | `.png` instead of a mesh | 415 |
| T5 | mesh without UVs | 422 `MESH_HAS_NO_UV` |
| — | feedback | 201 `recorded` |

## Scalability notes

- The model is ~0.5 MB and runs on CPU. Feature extraction is the slow part (pure numpy rasterisation,
  O(triangles)); for 50k+ triangle meshes run it in the Blender worker and call the API with `features`.
- The API is stateless except the in-memory job and feedback tables; run several uvicorn workers behind
  a load balancer and move jobs/feedback to a database or queue (Postgres, Redis) for production.

## Data

`python scripts/generate_data.py` recreates `data/assets.csv`. To use real data, export features with
`qa_model.features.load_mesh_features(path)` for each mesh, add `platform`, `asset_role`, `source` and a
TA label (`1` = not production ready), and train with `scripts/train.py`.
