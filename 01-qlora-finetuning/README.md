---

### 2. `01-qlora-finetuning/README.md`

```markdown
# 1. Fine-Tuning LLMs Using LoRA & QLoRA

## System Architecture & Evaluation Pipeline

```mermaid
flowchart TD
    A["400 Raw Seed Prompts"] --> B["Evol-Instruct + g++ Compiler Loop"]
    B -->|"258 Native Pass (64.5%) + 95 Repaired"| C["922 Verified C++ Training Pairs"]
    B -->|"47 Dropped (Unrecoverable)"| X["Discarded"]
    C --> D["Qwen2.5-Coder-7B Base (4-bit NF4 Frozen)"]
    E["Trainable LoRA Adapters (r=16, alpha=32, 0.53% Params)"] --> D
    D --> F["SFT Training on Tesla T4 (116 Steps, Loss 0.82 -> 0.30)"]
    F --> G["Blind 100-Prompt Held-Out g++ Benchmark"]
    G --> H["85.0% First-Pass Compilation (85/100)"]
    F --> I{"Deployment Strategy"}
    I -->|"Single Tenant"| J["Merge W0 + (alpha/r)BA (Zero Overhead)"]
    I -->|"Multi-Tenant"| K["Dynamic Multi-LoRA Serving (vLLM / LoRAX)"]
