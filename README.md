# LLM Systems Architecture — Design & Reference Implementations

Architectural designs, trade-off analyses, and working reference implementations covering four core production LLM engineering pillars:

| Module | Topic | Scope & Artifacts |
| :--- | :--- | :--- |
| **[`01-qlora-finetuning/`](./01-qlora-finetuning)** | Fine-Tuning LLMs with LoRA & QLoRA | Architecture diagram, QLoRA memory/rank mechanics, real `Qwen2.5-Coder-7B` training logs (`922` verified pairs), and blind 100-prompt `g++` benchmark (`64.5% -> 85.0%`) with failure root-cause analysis. |
| **[`02-speech-llm/`](./02-speech-llm)** | Converting a Text LLM to a Speech LLM | **Design-Only:** Comparison of Cascaded, Speech-Encoder + Projector, and Discrete Audio Token architectures, streaming/Indic code-switching trade-offs, and production recommendation. |
| **[`03-llm-gateway/`](./03-llm-gateway)** | Enterprise LLM Gateways | Architecture diagram, routing/failover, exact vs. semantic caching, FinOps governance, build-vs-buy analysis, and runnable FastAPI proxy (`gateway.py`). |
| **[`04-llm-guardrails/`](./04-llm-guardrails)** | Defense-in-Depth LLM Guardrails | Layered input/structural/output architecture, latency-cost comparison, streaming token-boundary analysis, and runnable FastAPI middleware (`guardrails.py`) wrapping Q3. |

## Quickstart (Combined Q3 Gateway + Q4 Guardrails Demo)

```bash
pip install -r requirements.txt

# Terminal 1: Start LLM Gateway (Port 8000)
uvicorn 03-llm-gateway.gateway:app --port 8000

# Terminal 2: Start Guardrails Middleware (Port 8001)
uvicorn 04-llm-guardrails.guardrails:app --port 8001
