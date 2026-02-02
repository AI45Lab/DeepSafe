"""
Minimal OpenAI-compatible router for serving multiple vLLM backends behind ONE port.

- Route decision: request JSON field "model"
- Upstreams configured via env var:
    UPSTREAMS="ModelA=http://127.0.0.1:21112/v1,ModelB=http://127.0.0.1:21113/v1"

Supports:
- GET  /health
- GET  /v1/models
- POST /v1/chat/completions
- POST /v1/completions
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

def _parse_upstreams(raw: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    raw = (raw or "").strip()
    if not raw:
        return mapping
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    for p in parts:
        if "=" not in p:
            raise ValueError(f"Bad UPSTREAMS entry (missing '='): {p}")
        k, v = p.split("=", 1)
        k = k.strip()
        v = v.strip().rstrip("/")
        if not k or not v:
            raise ValueError(f"Bad UPSTREAMS entry: {p}")
        mapping[k] = v
    return mapping

UPSTREAMS = _parse_upstreams(os.environ.get("UPSTREAMS", ""))
TIMEOUT_S = float(os.environ.get("ROUTER_TIMEOUT_S", "600"))

app = FastAPI()

@app.get("/health")
def health():
    return {"ok": True, "upstreams": list(UPSTREAMS.keys())}

@app.get("/v1/models")
def list_models():
    data = []
    for mid in sorted(UPSTREAMS.keys()):
        data.append({"id": mid, "object": "model", "created": 0, "owned_by": "vllm"})
    return {"object": "list", "data": data}

def _pick_upstream(payload: dict) -> Tuple[str, str]:
    model = payload.get("model")
    if not model:
        raise HTTPException(status_code=400, detail="Missing 'model' in request JSON.")
    if model not in UPSTREAMS:
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"Unknown model '{model}'.",
                "available_models": sorted(UPSTREAMS.keys()),
            },
        )
    return model, UPSTREAMS[model]

async def _proxy_json(request: Request, endpoint_suffix: str) -> Response:
    payload = await request.json()
    _, base = _pick_upstream(payload)
    stream = bool(payload.get("stream", False))

    url = f"{base}{endpoint_suffix}"

    headers = {}
                                                                          
    if "authorization" in request.headers:
        headers["authorization"] = request.headers["authorization"]

    async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
        if stream:
            upstream = await client.stream("POST", url, json=payload, headers=headers)
            return StreamingResponse(
                upstream.aiter_bytes(),
                status_code=upstream.status_code,
                media_type=upstream.headers.get("content-type", "application/octet-stream"),
            )
        else:
            resp = await client.post(url, json=payload, headers=headers)
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type=resp.headers.get("content-type", "application/json"),
            )

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await _proxy_json(request, "/chat/completions")

@app.post("/v1/completions")
async def completions(request: Request):
    return await _proxy_json(request, "/completions")

