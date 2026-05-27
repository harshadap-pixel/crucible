# 🔥 Crucible

> Local-first LLM evaluation — quality, RAG, agents, safety, and mechanism detection in one binary.

**No Python. No Node. No cloud account. Just [Ollama](https://ollama.com).**

Every other eval tool tests what your model **outputs**.  
Crucible also tests what your pipeline **is**.

```
crucible run --suite suites/default.toml

Run #a1b2c3d4  2026-05-26 14:23  model: llama3.1:8b
──────────────────────────────────────────────────────────────────────
TEST                          SCORE   STATUS    LATENCY   DELTA   FLAG
basic_factual                 1.000   ✅ PASS    312ms     +0.00
json_output                   1.000   ✅ PASS    289ms     +0.00
no_hallucination_on_unknown   0.920   ✅ PASS    401ms     -0.02
instruction_following         0.850   ✅ PASS    356ms     +0.01
refusal_jailbreak_basic       1.000   ✅ PASS    198ms     +0.00
──────────────────────────────────────────────────────────────────────
SUMMARY   5/5 passed   avg score: 0.954   ✓ no regressions
```

---

## Why Crucible?

| | Promptfoo | DeepEval | Ragas | **Crucible** |
|---|---|---|---|---|
| Single binary | ❌ Node | ❌ Python | ❌ Python | ✅ |
| Ollama-native | ✅ | ✅ | 🟡 | ✅ |
| RAG eval | 🟡 | ✅ | ✅ | ✅ |
| Agent + safety | ✅ | ✅ | ❌ | ✅ |
| **Mechanism detection** | ❌ | ❌ | ❌ | ✅ |
| **MoE detection** | ❌ | ❌ | ❌ | ✅ |
| **RAG fallback probes** | ❌ | ❌ | ❌ | ✅ |
| Regression tracking | 🟡 | ❌ | ❌ | ✅ |
| Embedded SQLite | ✅ | ❌ | ❌ | ✅ |

---

## Install

### From GitHub Releases (recommended)

```bash
# macOS Apple Silicon
curl -L https://github.com/harshadap-pixel/crucible/releases/latest/download/crucible-aarch64-macos \
  -o /usr/local/bin/crucible && chmod +x /usr/local/bin/crucible

# macOS Intel
curl -L https://github.com/harshadap-pixel/crucible/releases/latest/download/crucible-x86_64-macos \
  -o /usr/local/bin/crucible && chmod +x /usr/local/bin/crucible

# Linux x86_64
curl -L https://github.com/harshadap-pixel/crucible/releases/latest/download/crucible-x86_64-linux \
  -o /usr/local/bin/crucible && chmod +x /usr/local/bin/crucible
```

### From source

```bash
cargo install --git https://github.com/harshadap-pixel/crucible
```

---

## Quick Start

```bash
# 1. Start Ollama with any model
ollama serve
ollama pull llama3.1:8b

# 2. Run the default suite
crucible run

# 3. Set this run as your regression baseline
crucible baseline set

# 4. Make a change (swap model, tweak prompt, update RAG config...)
# 5. Run again — Crucible will flag regressions automatically
crucible run
```

---

## Commands

```
crucible run       [--suite <path>] [--model <name>] [--n-runs <N>]
crucible detect    --model <name>   [--pipeline-config <path>]
crucible baseline  set | show
crucible compare   <run-id-a> <run-id-b>
crucible report    [--last N] [--suite <name>]
crucible status
```

---

## Test Suite Format (TOML)

```toml
[suite]
name                 = "My RAG Pipeline"
model                = "llama3.1:8b"
judge                = "llama3.3:70b"     # larger model as LLM-as-judge
concurrency          = 4
regression_threshold = 0.05               # flag if score drops >5%

[[tests]]
name    = "grounded_answer"
prompt  = "Based only on the context, what year was X built?"
context = ["X was built in 1889."]

  [[tests.assert]]
  type  = "contains"
  value = "1889"

  [[tests.assert]]
  type      = "llm_judge"
  rubric    = "Answer must be grounded in the context. No hallucination."
  threshold = 0.85
```

### Assertion types

| Type | Description |
|---|---|
| `contains` | Output must include this substring |
| `not_contains` | Output must NOT include this substring |
| `regex` | Output must match this pattern |
| `exact_match` | Output must equal this string exactly |
| `json_schema` | Output must be valid JSON matching this schema |
| `semantic` | Cosine similarity ≥ threshold (via Ollama embeddings) |
| `llm_judge` | LLM grades output against a rubric (0.0–1.0) |
| `refusal_check` | Output must look like a refusal (safety tests) |
| `tool_not_called` | Agent must not call this tool |

---

## Mechanism Detection

```bash
crucible detect --model mixtral:8x7b --pipeline-config ./rag/config.yaml

MODEL METADATA
──────────────────────────────────────────────────────────────
Architecture      mixtral
Attention type    GQA (Grouped Query Attention)
MoE               YES — 8 total experts / 2 active per token (75.0% sparse)
Context length    32768 tokens
Parameters        46.7B
Quantization      Q4_K_M

KV CACHE
  Cold latency  312ms   Warm latency  134ms   Speedup  2.33x
  Status        LIKELY ACTIVE

PIPELINE MECHANISMS (from config)
  HNSWlib                        DETECTED
  Chunking (recursive)           DETECTED
  Query expansion                DETECTED
  HyDE                           not found
  Reranking                      not found

⚠ MoE model detected — recommend --n-runs 5+ for stable scores
```

---

## Built-in Suites

| Suite | Path | Coverage |
|---|---|---|
| Default eval | `suites/default.toml` | Factual, JSON, instruction-following, basic safety |
| OWASP LLM01 | `suites/safety/owasp_llm01_injection.toml` | Prompt injection (direct + indirect) |
| RAG faithfulness | `suites/rag/faithfulness.toml` | Groundedness, no hallucination, context relevance |
| RAG fallback | `suites/rag/fallback_chain.toml` | Typo robustness, graceful decline, semantic normalization |

---

## Regression Workflow

```bash
# Day 1: establish baseline
crucible run --suite suites/rag/faithfulness.toml
crucible baseline set

# Day 7: after updating your RAG pipeline
crucible run --suite suites/rag/faithfulness.toml
# → Crucible automatically diffs against baseline
# → Flags score drift (>5% drop) and pass→fail flips

# Explicit comparison between two specific runs
crucible compare a1b2c3d4 e5f6g7h8
```

---

## Roadmap

- **v0.1** (now) — Core eval runner, mechanism detection, SQLite regression, built-in suites
- **v0.2** — RAG IR metrics (Recall@k, MRR), long-context needle tests, latency as first-class metric
- **v0.3** — Full OWASP LLM Top 10 probes, agent multi-turn loop eval, LlamaGuard integration
- **v0.4** — MMLU/HumanEval benchmark runners, statistical significance, cost tracking

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
