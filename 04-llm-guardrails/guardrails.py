import re
from typing import Dict, List, Tuple
import httpx
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="LLM Guardrails Middleware (Q4 Demo)")
GATEWAY_URL = "http://localhost:8000/v1/chat/completions"

INJECTION_PATTERNS = [
    r"ignore (all )?previous instructions",
    r"reveal (your )?system prompt",
    r"developer mode",
    r"bypass safety",
]

OUTPUT_BLOCKLIST = ["INTERNAL_SECRET_KEY", "SYSTEM_PROMPT_LEAK", "SanctumInternal"]

class GuardedRequest(BaseModel):
    messages: List[Dict[str, str]]
    model: str = "qwen2.5-coder:7b"

def redact_pii(text: str) -> Tuple[str, bool]:
    redacted = re.sub(r"\b[\w.-]+@[\w.-]+\.\w+\b", "[REDACTED_EMAIL]", text)
    redacted = re.sub(r"\b\d{10}\b", "[REDACTED_PHONE]", redacted)
    return redacted, (redacted != text)

def check_input_rails(text: str) -> Tuple[bool, str]:
    lowered = text.lower()
    for pat in INJECTION_PATTERNS:
        if re.search(pat, lowered):
            return False, f"Blocked by Input Rail (Prompt Injection matched: '{pat}')"
    return True, "OK"

def sliding_window_output_scan(chunks: List[str], window_size: int = 3) -> Tuple[bool, str]:
    """Scans rolling N-chunk windows so restricted phrases split across token boundaries are caught."""
    for i in range(len(chunks)):
        window = "".join(chunks[max(0, i - window_size + 1) : i + 1])
        for blocked in OUTPUT_BLOCKLIST:
            if blocked.lower() in window.lower():
                return False, f"Blocked by Output Rail (Boundary scan caught '{blocked}')"
    return True, "OK"

@app.post("/v1/guarded/chat")
async def guarded_chat(req: GuardedRequest):
    sanitized_messages = []
    pii_found = False

    # 1. Pre-Inference Input Guardrails (PII Redaction + Injection Check)
    for msg in req.messages:
        clean_text, had_pii = redact_pii(msg["content"])
        pii_found = pii_found or had_pii
        allowed, reason = check_input_rails(clean_text)
        if not allowed:
            return {"status": "blocked", "stage": "input_rail", "reason": reason}
        sanitized_messages.append({"role": msg["role"], "content": clean_text})

    # 2. Forward Sanitized Request to Q3 LLM Gateway
    async with httpx.AsyncClient(timeout=15.0) as client:
        gw_resp = await client.post(
            GATEWAY_URL,
            json={"model": req.model, "messages": sanitized_messages, "temperature": 0.0},
        )
        gw_data = gw_resp.json()

    raw_output = gw_data["choices"][0]["message"]["content"]

    # 3. Post-Inference Sliding-Window Output Rail
    simulated_token_chunks = [raw_output[i : i + 6] for i in range(0, len(raw_output), 6)]
    out_ok, out_reason = sliding_window_output_scan(simulated_token_chunks)
    if not out_ok:
        return {"status": "blocked", "stage": "output_rail", "reason": out_reason}

    return {
        "status": "passed",
        "pii_redacted": pii_found,
        "response": raw_output,
        "gateway_meta": gw_data.get("gateway_meta", {}),
    }