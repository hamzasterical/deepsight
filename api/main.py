from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.detect import router as detect_router
from src.inference.predictor import Predictor
from src.models.dual_branch import DualBranchModel

_predictor: Optional[Predictor] = None


def get_predictor() -> Predictor:
    global _predictor
    if _predictor is None:
        import torch
        model = DualBranchModel()
        model.eval()
        _predictor = Predictor(model)
    return _predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    _warmup()
    yield


def _warmup():
    predictor = get_predictor()
    import numpy as np
    dummy = np.zeros((224, 224, 3), dtype=np.uint8)
    try:
        predictor.predict(dummy)
    except Exception:
        pass


app = FastAPI(
    title="DeepSight — Image Forgery Detection",
    description="Real-time passive image forgery detection using dual-branch CNN",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(detect_router, prefix="/api/v1", tags=["detection"])


@app.get("/health")
async def health():
    return {"status": "ok"}
