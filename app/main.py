"""FastAPI service for the digit / not-a-digit classifier.

    uvicorn app.main:app --reload

Endpoints
    GET  /                 the drawing canvas UI
    GET  /api/health       model status, thresholds, parameter count
    POST /api/predict      {"image": "data:image/png;base64,..."} or a file upload
    GET  /api/sample/{kind} a generated example (star_field, scribble, ...)

The last one exists for demos: it lets someone click "star field" and watch the
model reject it, without having to find an out-of-distribution image first.
"""

from __future__ import annotations

import base64
import io
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from digitood.data.negatives import SYNTHETIC_GENERATORS
from digitood.data.glyphs import render_glyph
from digitood.inference.predictor import Predictor
from digitood.inference.preprocess import prepare_data_url, prepare_bytes

STATIC_DIR = Path(__file__).parent / "static"
CHECKPOINT = os.getenv("MODEL_PATH", "models/digit_ood_cnn.pt")
MAX_UPLOAD_BYTES = 4 * 1024 * 1024

# Populated by the lifespan handler below. Loading the checkpoint once at
# startup rather than per request is the difference between a ~5 ms response
# and a ~2 s one.
state: dict[str, Any] = {"predictor": None, "error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        state["predictor"] = Predictor(CHECKPOINT)
        print(f"loaded model from {CHECKPOINT}")
    except FileNotFoundError:
        # Serve the UI anyway and report the problem through /api/health, so
        # "I forgot to train" is an obvious message instead of a stack trace.
        state["error"] = (
            f"no checkpoint at {CHECKPOINT} -- run "
            f"`python -m digitood.training.train` first"
        )
        print(f"WARNING: {state['error']}")
    yield


app = FastAPI(title="0-9 Digit Classification (+ unclassifiable)",
              version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class PredictRequest(BaseModel):
    image: str  # a data: URL from the browser canvas


def _require_model() -> Predictor:
    if state["predictor"] is None:
        raise HTTPException(status_code=503, detail=state["error"] or "model not loaded")
    return state["predictor"]


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> JSONResponse:
    predictor = state["predictor"]
    if predictor is None:
        return JSONResponse({"status": "no_model", "detail": state["error"]},
                            status_code=503)
    return JSONResponse({
        "status": "ok",
        "parameters": predictor.model.num_parameters(),
        "min_confidence": round(predictor.min_confidence, 4),
        "energy_threshold": (round(predictor.energy_threshold, 4)
                             if predictor.energy_threshold is not None else None),
        "test_metrics": predictor.metadata.get("test_metrics"),
    })


@app.post("/api/predict")
def predict(payload: PredictRequest) -> dict[str, Any]:
    predictor = _require_model()
    try:
        prepared = prepare_data_url(payload.image)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"could not decode image: {exc}")
    return predictor.predict_prepared(prepared).to_dict()


@app.post("/api/predict-file")
async def predict_file(file: UploadFile = File(...)) -> dict[str, Any]:
    predictor = _require_model()
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="file larger than 4 MB")
    try:
        prepared = prepare_bytes(data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"could not read image: {exc}")
    return predictor.predict_prepared(prepared).to_dict()


@app.get("/api/sample/{kind}")
def sample(kind: str) -> dict[str, str]:
    """Generate one example of a given negative family, as a PNG data URL."""
    rng = np.random.default_rng()
    if kind == "glyph":
        arr, _ = render_glyph(rng)
    elif kind in SYNTHETIC_GENERATORS:
        arr = SYNTHETIC_GENERATORS[kind][0](rng)
    else:
        raise HTTPException(
            status_code=404,
            detail=f"unknown sample kind; try one of: glyph, "
                   f"{', '.join(SYNTHETIC_GENERATORS)}",
        )

    # Upscale so the 28x28 array is visible on the canvas without the browser
    # blurring it into something the model never sees.
    img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), mode="L")
    img = img.resize((280, 280), Image.Resampling.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return {"kind": kind,
            "image": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}
