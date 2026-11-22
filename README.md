# AI Agent 3D Production Studio — UV/Readiness QA Model

BD532 Term Project (AI Independent Study) · Kulchaya Sinsiriwanich 68130014

ทีม AI Agent 8 ตำแหน่งที่สร้างตัวละคร 3D จนใช้งานได้ใน Unreal Engine 5
Repository นี้คือ **ส่วน ML ของระบบ (Section 3–4 ของรายงาน)**: โมเดลที่ตัดสินว่าโมเดล 3D
"พร้อมเข้า Engine" หรือไม่ โดยเน้นปัญหา UV ที่ Tripo AI ยังทำได้ไม่ดี พร้อมอธิบายเหตุผลด้วย SHAP
และบอกว่าควรส่งงานกลับให้ Agent ตัวไหนแก้

```
Customer -> Manager -> 2D Art Director -> 3D Modeler A (Lead) -> 3D Modeler B (Image-to-3D)
         -> 3D Modeler C (UV + Texture) -> [UV/Readiness QA model]  <- this repo
         -> Technical Artist A (Unreal import) -> Manager review -> Customer
```

> **Proof of concept:** training data in `data/assets.csv` is **synthetic** (`qa_model/data.py`),
> generated from the SPEC thresholds with simulated Technical-Artist disagreement. The feature
> extractor is real and works on .glb/.gltf/.obj files; replace the synthetic rows with features
> extracted from real Tripo outputs + TA labels in the IS phase. The agent pipeline behind
> `/v1/jobs` is simulated.

## Project layout

| Path | What it is |
| --- | --- |
| `qa_model/features.py` | Measures a mesh: UV islands, overlap %, stretch, texel-density CV, seams, coverage, quad ratio, non-manifold edges |
| `qa_model/data.py` | Synthetic dataset (4 sources: `tripo_raw`, `tripo_fixed`, `artist`, `objaverse`) + SPEC labelling |
| `qa_model/model.py` | The 4 model versions compared in MLflow |
| `qa_model/explain.py`, `predict.py` | SHAP reasons in Thai, issues, next agent to call |
| `qa_model/config.py` | SPEC thresholds, triangle/texture budgets per platform and asset role |
| `scripts/` | `generate_data`, `eda`, `train`, `explain_model`, `make_samples` |
| `api/main.py` | FastAPI service (`/v1/assess`, `/v1/jobs`, `/v1/health`) |
| `postman/` | Postman collection for test scenarios T1–T5 |
| `tests/` | pytest: feature extractor, data, API scenarios |
| `samples/` | Test meshes: `char_fixed.glb`, `char_tripo_raw.glb`, `no_uv.glb` |
| `reports/` | EDA plots, model comparison, confusion matrices, SHAP plots, fairness, 3 explained cases |
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
python scripts/eda.py               # 3.1 data exploration -> reports/eda_*.png
python scripts/train.py             # 3.1 + 3.3 train 4 versions, log to MLflow, save best model
python scripts/explain_model.py     # 3.2 fairness, 3.4 SHAP summary, 3.5 three explained cases
python scripts/make_samples.py      # sample meshes for the API tests
pytest -q                           # unit + API tests (18 tests)
```

MLflow UI (3.3):

```powershell
mlflow ui --backend-store-uri sqlite:///mlruns.db
```

then open http://127.0.0.1:5000 and the experiment `uv-readiness-qa`.

## Results (synthetic data, test split 20% = 280 assets)

| Version | Features | Threshold | Test Recall (FAIL) | Test Precision | Test F1 | ROC AUC |
| --- | --- | --- | --- | --- | --- | --- |
| v1 Logistic Regression | mesh only | 0.23 | 0.989 | 0.684 | 0.809 | 0.749 |
| v2 Random Forest | mesh only | 0.23 | 0.888 | 0.746 | 0.811 | 0.763 |
| v3 XGBoost | mesh only | 0.29 | 0.957 | 0.729 | 0.828 | 0.766 |
| **v4 XGBoost + context** | mesh + platform/role budgets | 0.24 | **0.936** | **0.834** | **0.882** | **0.808** |

**Selected: v4.** Rule: best cross-validated F1 among versions with cross-validated Recall(FAIL) ≥ 0.90
(the report's goal: never let a bad model reach the customer). Context matters because the same mesh can
pass on PC and fail on Mobile: `tri_budget_ratio` is the strongest feature in the SHAP summary.
Report goals met: Recall ≥ 0.90 and F1 ≥ 0.85. Not met yet: Cohen's kappa vs labels 0.59 (goal 0.7) —
expected with simulated label noise; to be re-measured with real TA labels.

Fairness (3.2, equal opportunity = Recall on FAIL per source): artist 0.87, objaverse 1.00,
tripo_fixed 0.78, tripo_raw 1.00 → gap 0.22. The model misses more bad assets among already-fixed
Tripo models, whose problems are subtler. Mitigation: per-source threshold or more labelled
`tripo_fixed` examples. See `reports/fairness.csv`.

## API (Section 4)

```powershell
uvicorn api.main:app --reload
```

Swagger docs: http://127.0.0.1:8000/docs

### `POST /v1/assess`

Request (send `file_url` **or** pre-computed `features`):

```json
{
  "asset_id": "char_tripo_raw",
  "file_url": "samples/char_tripo_raw.glb",
  "context": { "platform": "PC", "engine": "UE5", "asset_role": "hero_character", "target_fps": 60 }
}
```

`platform`: `PC` | `Console` | `Mobile` · `asset_role`: `hero_character` | `npc_character` | `prop`.
`file_url` is a local path (relative to this folder or absolute) to `.glb`, `.gltf` or `.obj`.

Response 200 (shortened):

```json
{
  "asset_id": "char_tripo_raw",
  "status": "FAIL",
  "readiness_score": 13,
  "fail_probability": 0.8731,
  "model_version": "v4_xgboost_context",
  "issues": [
    { "code": "UV_OVERLAP", "feature": "uv_overlap_pct", "severity": "high", "value": 92.9635, "limit": 1.0,
      "agent": "3d_modeler_c", "task": "re_unwrap" },
    { "code": "UV_FRAGMENTED", "feature": "islands_per_1k_tris", "severity": "high", "value": 1000.0, "limit": 8.0,
      "agent": "3d_modeler_c", "task": "re_unwrap" }
  ],
  "reasons": [
    { "feature": "seam_ratio", "value": 1.0, "shap": 0.6983, "text": "มี Seam มากเกินไป" },
    { "feature": "uv_island_count", "value": 2304.0, "shap": 0.6389, "text": "จำนวน UV island สูง" },
    { "feature": "uv_overlap_pct", "value": 92.9635, "shap": 0.5648, "text": "UV ทับกัน (overlap)" }
  ],
  "next_action": { "agent": "3d_modeler_c", "task": "re_unwrap" },
  "needs_human_review": false,
  "features": { "...": "all measured values" },
  "latency_ms": 1039.3
}
```

The first request takes about 1 s because the SHAP explainer is built; later requests are faster.

Errors always look like `{"error": {"code": "...", "message": "...", "details": ...}}`:

| Status | Code | When |
| --- | --- | --- |
| 400 | `INVALID_PAYLOAD` | missing/invalid field (e.g. `context.platform`), or neither `file_url` nor `features` |
| 404 | `FILE_NOT_FOUND` | `file_url` does not exist |
| 415 | `UNSUPPORTED_FORMAT` | not .glb/.gltf/.obj (FBX: convert in the Blender worker first) |
| 422 | `MESH_HAS_NO_UV` / `MESH_UNREADABLE` | mesh cannot be assessed |
| 503 | `MODEL_NOT_READY` | `models/qa_model.joblib` missing |

### `POST /v1/jobs` → 202, `GET /v1/jobs/{job_id}`

Starts a production job and returns the stage of each agent for the Production Tracker screen.
In this proof of concept the stages advance on a timer (3 s each); no real agents are called.

### Test scenarios (4.2)

Import `postman/AI-Agent-3D-Production.postman_collection.json` into Postman, start the API, run the
collection. The same scenarios run in `tests/test_api.py`.

| # | Scenario | Expected |
| --- | --- | --- |
| T1 | raw Tripo-like mesh (every triangle its own overlapping island) | 200 FAIL, `UV_FRAGMENTED`, next agent `3d_modeler_c` |
| T2 | same mesh after a clean unwrap | 200 PASS, score ≥ 80 |
| T3 | `context.platform` missing | 400 naming `context.platform` |
| T4 | `.png` instead of a mesh | 415 |
| T5 | mesh without UVs | 422 `MESH_HAS_NO_UV` |

## Scalability notes

- The model is ~0.5 MB and runs on CPU; one assessment takes milliseconds. Feature extraction is the
  slow part (pure numpy rasterisation, O(triangles)); for 50k+ triangle meshes move it to a worker queue
  (Blender worker in the report architecture) and call `/v1/assess` with `features`.
- The API is stateless except the in-memory job table; run several uvicorn workers behind a load
  balancer and move jobs to a database / queue (Redis, Postgres) for production.

## Data

`python scripts/generate_data.py` recreates `data/assets.csv`. To use real data, export features with
`qa_model.features.load_mesh_features(path)` for each mesh, add `platform`, `asset_role`, `source` and a
TA label (`1` = not production ready), and train with `scripts/train.py`.
