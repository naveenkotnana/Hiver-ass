"""FastAPI entrypoint: POST /predict

    uvicorn app:app --reload
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.agent.support_agent import SupportAgent

app = FastAPI(title="Hiver Apple Support Agent", version="1.0.0")


class PredictRequest(BaseModel):
    message: str = Field(..., min_length=1)
    context: str = ""


class EvidenceItem(BaseModel):
    conversation_id: str | None = None
    customer_message: str | None = None
    brand_response: str | None = None
    similarity: float | None = None
    intent: str | None = None


class PredictResponse(BaseModel):
    intent: str
    intent_confidence: float
    decision: str
    reason: str
    reply: str
    evidence: list[dict[str, Any]]


@lru_cache(maxsize=1)
def get_agent() -> SupportAgent:
    try:
        return SupportAgent()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Models not trained. Run python scripts/run_demo.py") from exc


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest) -> PredictResponse:
    result = get_agent().handle(body.message, context=body.context)
    return PredictResponse(
        intent=result["intent"],
        intent_confidence=float(result["intent_confidence"]),
        decision=result["decision"],
        reason=result["reason"],
        reply=result["draft_reply"],
        evidence=result.get("evidence") or [],
    )
