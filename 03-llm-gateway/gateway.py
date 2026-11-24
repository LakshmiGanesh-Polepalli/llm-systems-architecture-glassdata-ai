import hashlib
import json
import os
import time
import uuid
from typing import Any, Dict, List
import httpx
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="LLM Gateway (Q3 Demo)")

EXACT_CACHE: Dict[str, Dict[str, Any]] = {}
BUDGET_LEDGER: Dict[str, float] = {"default-team": 25.00}

class ChatRequest(BaseModel):
    team_id: str = "default-team"
    model: str = "qwen2.5-coder:7b"
    messages: List[Dict[str, str]]
    temperature: float = 0.0

async def call_ollama(model: str, messages: List[Dict[str, str]]) -> str:
    async with httpx.AsyncClient(timeout=2.5) as client:
        resp = await client.post(
            "http://localhost:11434/api/chat",
            json={"model": model, "messages": messages, "stream": False},
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]

async def call_cloud_fallback(messages: List[Dict[str, str]]) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        # Deterministic fallback stub so demo runs offline without paid keys
        return f"[Fallback:openrouter/llama-3.1-8b] Response to: {messages[-1]['content']}"
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": "meta-llama/llama-3.1-8b-instruct", "messages": messages},
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    req_id = str(uuid.uuid4())[:8]
    t0 = time.perf_counter()

    # 1. Exact SHA-256 Cache Lookup
    cache_key = hashlib.sha256(
        json.dumps({"m": req.model, "msgs": req.messages}, sort_keys=True).encode()
    ).hexdigest()

    if req.temperature == 0.0 and cache_key in EXACT_CACHE:
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        log = {"req_id": req_id, "provider": "cache", "cache_hit": True, "latency_ms": latency_ms, "cost_usd": 0.0}
        print(json.dumps(log))
        return {**EXACT_CACHE[cache_key], "gateway_meta": log}

    # 2. Priority Routing: Local Ollama -> Cloud Fallback
    fallback_triggered = False
    try:
        output_text = await call_ollama(req.model, req.messages)
        provider = f"ollama/{req.model}"
        cost_usd = 0.0
    except Exception:
        fallback_triggered = True
        output_text = await call_cloud_fallback(req.messages)
        provider = "openrouter/fallback"
        cost_usd = 0.0002

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    est_tokens = sum(len(m["content"].split()) for m in req.messages) + len(output_text.split())
    BUDGET_LEDGER[req.team_id] = round(BUDGET_LEDGER.get(req.team_id, 10.0) - cost_usd, 6)

    meta = {
        "req_id": req_id,
        "provider": provider,
        "fallback_triggered": fallback_triggered,
        "cache_hit": False,
        "latency_ms": latency_ms,
        "est_tokens": est_tokens,
        "cost_usd": cost_usd,
        "remaining_budget_usd": BUDGET_LEDGER[req.team_id],
    }
    print(json.dumps(meta))  # Structured observability log

    payload = {"choices": [{"message": {"role": "assistant", "content": output_text}}], "gateway_meta": meta}
    if req.temperature == 0.0:
        EXACT_CACHE[cache_key] = payload
    return payload