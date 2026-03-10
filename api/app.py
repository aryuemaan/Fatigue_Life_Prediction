"""
app.py — FastAPI production backend
====================================
Endpoints
---------
GET  /                   — Landing page HTML
GET  /health             — Health + model status
GET  /model/info         — Architecture, training metrics, label stats
POST /predict            — Single image → full prediction JSON + annotated PNG
POST /predict/batch      — Multiple images → list of predictions
POST /predict/gradcam    — Single image → prediction + GradCAM heatmap PNG
GET  /docs               — Swagger UI
GET  /redoc              — ReDoc UI

Run
---
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
"""

import cv2
import numpy as np
import base64
import sys
import os
import json
import time
import io
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ══════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ══════════════════════════════════════════════════════════

class PredictionOut(BaseModel):
    sample_id:          str
    log10_Nf:           float = Field(description="log₁₀(cycles to failure)")
    Nf_cycles:          float = Field(description="Predicted cycles to failure")
    log10_Nf_lower:     float = Field(description="95% CI lower bound")
    log10_Nf_upper:     float = Field(description="95% CI upper bound")
    std:                float = Field(description="Prediction std deviation (log scale)")
    risk_category:      str   = Field(description="LOW | MEDIUM | HIGH | CRITICAL")
    inference_ms:       float
    annotated_image_b64: Optional[str] = Field(None, description="Base64-encoded annotated PNG")


class BatchOut(BaseModel):
    results:      List[PredictionOut]
    n_processed:  int
    total_ms:     float


class ModelInfoOut(BaseModel):
    architecture:  str
    label_mean:    float
    label_std:     float
    is_mock:       bool
    checkpoint:    str
    description:   str


# ══════════════════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════════════════

app = FastAPI(
    title="Steel Fatigue Life Predictor",
    description=(
        "CNN-based computer vision API for predicting fatigue life of "
        "lightweight alloy steels from optical microscopy images."
    ),
    version="2.0.0",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


# ══════════════════════════════════════════════════════════
# STARTUP — load model
# ══════════════════════════════════════════════════════════

ENGINE = None

@app.on_event("startup")
async def startup():
    global ENGINE
    from inference.inference_engine import InferenceEngine
    ckpt = os.getenv("MODEL_PATH", "outputs/best_resnet50.pth")
    ENGINE = InferenceEngine(ckpt)
    import logging
    logging.getLogger("uvicorn.access").info(f"Engine ready, mock={ENGINE._mock_mode}")


# ══════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════

def decode_upload(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(422, "Cannot decode image. Upload a valid PNG/JPG/TIF.")
    return img


def img_to_b64(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode()


def result_to_out(res, include_img: bool = True) -> PredictionOut:
    b64 = img_to_b64(res.annotated_img) if (include_img and res.annotated_img is not None) else None
    return PredictionOut(
        sample_id          = res.sample_id,
        log10_Nf           = res.log10_Nf,
        Nf_cycles          = res.Nf_cycles,
        log10_Nf_lower     = res.log10_Nf_lower,
        log10_Nf_upper     = res.log10_Nf_upper,
        std                = res.std,
        risk_category      = res.risk_category,
        inference_ms       = res.inference_ms,
        annotated_image_b64= b64,
    )


# ══════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    mock = ENGINE._mock_mode if ENGINE else True
    return {"status": "ok", "model_loaded": ENGINE is not None,
            "mock_mode": mock}


@app.get("/model/info", response_model=ModelInfoOut)
async def model_info():
    if ENGINE is None:
        raise HTTPException(503, "Engine not initialised")
    return ModelInfoOut(
        architecture = ENGINE.arch,
        label_mean   = ENGINE.label_stats["mean"],
        label_std    = ENGINE.label_stats["std"],
        is_mock      = ENGINE._mock_mode,
        checkpoint   = os.getenv("MODEL_PATH", "outputs/best_resnet50.pth"),
        description  = (
            "CNN regression model (ResNet50/VGG16/Custom) predicting log₁₀(N_f) "
            "from steel optical microscopy images. Includes epistemic uncertainty "
            "via Gaussian NLL loss head."
        ),
    )


@app.post("/predict", response_model=PredictionOut)
async def predict(
    file:           UploadFile = File(...),
    sample_id:      Optional[str] = Query(None),
    return_gradcam: bool = Query(True,  description="Include GradCAM overlay"),
    return_image:   bool = Query(True,  description="Include annotated image in response"),
):
    """
    Upload a steel microscopy image.
    Returns fatigue life prediction with optional GradCAM visualisation.
    """
    if ENGINE is None:
        raise HTTPException(503, "Model engine not ready")
    raw = decode_upload(await file.read())
    sid = sample_id or Path(file.filename or "img").stem
    res = ENGINE.predict(raw, return_gradcam=return_gradcam,
                          annotate=return_image, sample_id=sid)
    return result_to_out(res, include_img=return_image)


@app.post("/predict/batch", response_model=BatchOut)
async def predict_batch(
    files: List[UploadFile] = File(...),
    return_images: bool = Query(False),
):
    """Predict fatigue life for multiple images."""
    if ENGINE is None:
        raise HTTPException(503, "Model engine not ready")
    t0      = time.time()
    results = []
    for f in files:
        try:
            raw = decode_upload(await f.read())
            sid = Path(f.filename or "img").stem
            res = ENGINE.predict(raw, return_gradcam=False,
                                  annotate=return_images, sample_id=sid)
            results.append(result_to_out(res, include_img=return_images))
        except Exception as e:
            results.append(PredictionOut(
                sample_id=f.filename or "?",
                log10_Nf=0, Nf_cycles=0, log10_Nf_lower=0,
                log10_Nf_upper=0, std=0,
                risk_category="ERROR", inference_ms=0,
                annotated_image_b64=None,
            ))
    return BatchOut(results=results, n_processed=len(results),
                    total_ms=round((time.time()-t0)*1000, 1))


@app.post("/predict/gradcam")
async def predict_gradcam(file: UploadFile = File(...)):
    """
    Returns the annotated image directly as a PNG (no JSON wrapper).
    Useful for embedding in web pages: <img src="/predict/gradcam">.
    """
    if ENGINE is None:
        raise HTTPException(503, "Model engine not ready")
    raw = decode_upload(await file.read())
    res = ENGINE.predict(raw, return_gradcam=True, annotate=True)
    if res.annotated_img is None:
        raise HTTPException(500, "Annotation failed")
    ok, buf = cv2.imencode(".png", res.annotated_img)
    return Response(content=buf.tobytes(), media_type="image/png")


# ══════════════════════════════════════════════════════════
# LANDING PAGE
# ══════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def landing():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fatigue Life Predictor API</title>
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow:wght@300;600&display=swap" rel="stylesheet">
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  :root{--g:#00c896;--bg:#0b0d12;--panel:#13161f;--border:#1e2436;--text:#ccd6f0;--dim:#6b7a9f}
  body{font-family:'Barlow',sans-serif;background:var(--bg);color:var(--text);min-height:100vh;
       padding:40px 20px;background-image:radial-gradient(ellipse at 20% 20%,rgba(0,200,150,.04) 0%,transparent 60%)}
  h1{font-family:'Share Tech Mono',monospace;font-size:2rem;color:var(--g);letter-spacing:.08em;margin-bottom:6px}
  .sub{color:var(--dim);font-weight:300;margin-bottom:40px;font-size:.95rem}
  .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:16px;max-width:960px}
  .card{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:24px;
        transition:border-color .2s,transform .2s}
  .card:hover{border-color:var(--g);transform:translateY(-2px)}
  .method{font-family:'Share Tech Mono',monospace;font-size:.7rem;padding:3px 8px;
          border-radius:3px;margin-right:8px;letter-spacing:.05em}
  .get{background:#0d3a2a;color:#00c896} .post{background:#2a1a0d;color:#ff9040}
  .ep{font-family:'Share Tech Mono',monospace;font-size:.88rem;color:var(--g);margin:10px 0 6px}
  .desc{font-size:.83rem;color:var(--dim);line-height:1.5}
  .badge{display:inline-block;padding:4px 12px;border-radius:20px;font-size:.75rem;
         font-weight:600;letter-spacing:.05em;margin:2px}
  .low{background:#0a2a14;color:#00d066;border:1px solid #00d06640}
  .med{background:#2a200a;color:#f0b000;border:1px solid #f0b00040}
  .high{background:#2a100a;color:#ff6030;border:1px solid #ff603040}
  .crit{background:#2a0a0a;color:#ff2020;border:1px solid #ff202040}
  .links{margin-top:40px;display:flex;gap:16px;flex-wrap:wrap}
  .links a{color:var(--g);text-decoration:none;font-size:.9rem;border:1px solid #00c89630;
           padding:8px 20px;border-radius:4px;font-family:'Share Tech Mono',monospace;
           transition:background .2s}
  .links a:hover{background:#00c89615}
  hr{border:none;border-top:1px solid var(--border);margin:30px 0}
</style>
</head>
<body>
<h1>⚙ FATIGUE LIFE PREDICTOR</h1>
<p class="sub">CNN Computer Vision API · Lightweight Alloy Steel Microscopy · v2.0</p>
<div class="grid">
  <div class="card">
    <span class="method post">POST</span>
    <div class="ep">/predict</div>
    <div class="desc">Upload a microscopy image. Returns log₁₀(N<sub>f</sub>), cycles to failure, 95% CI, risk category, and optional GradCAM-annotated PNG.</div>
  </div>
  <div class="card">
    <span class="method post">POST</span>
    <div class="ep">/predict/batch</div>
    <div class="desc">Submit multiple images in a single request. Returns ordered prediction list with per-sample metadata.</div>
  </div>
  <div class="card">
    <span class="method post">POST</span>
    <div class="ep">/predict/gradcam</div>
    <div class="desc">Returns the annotated image directly as a PNG stream — embed directly in &lt;img&gt; tags.</div>
  </div>
  <div class="card">
    <span class="method get">GET</span>
    <div class="ep">/model/info</div>
    <div class="desc">Current model architecture, training label statistics, and mock-mode status.</div>
  </div>
  <div class="card">
    <span class="method get">GET</span>
    <div class="ep">/health</div>
    <div class="desc">Service health check. Returns model load status.</div>
  </div>
</div>
<hr>
<p style="color:var(--dim);font-size:.85rem;margin-bottom:12px">Risk classification thresholds:</p>
<span class="badge low">LOW &nbsp; log(Nf) &gt; 6.5</span>
<span class="badge med">MEDIUM &nbsp; 5.5 – 6.5</span>
<span class="badge high">HIGH &nbsp; 4.5 – 5.5</span>
<span class="badge crit">CRITICAL &nbsp; &lt; 4.5</span>
<div class="links">
  <a href="/docs">📖 Swagger UI</a>
  <a href="/redoc">📄 ReDoc</a>
  <a href="/health">💚 Health</a>
  <a href="/model/info">🧠 Model Info</a>
</div>
</body>
</html>"""
