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
| **Full OWASP LLM Top 10** | 🟡 partial | 🟡 partial | ❌ | ✅ all 10 |
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
crucible run       --models model1,model2,model3    # leaderboard mode
crucible detect    --model <name>   [--pipeline-config <path>]
crucible baseline  set | show
crucible compare   <run-id-a> <run-id-b>
crucible report    [--last N] [--suite <name>]
crucible status
```

---

## Model Leaderboard

Compare any number of models on the same suite in one command:

```bash
crucible run --suite suites/default.toml \
  --models llama3.1:8b,mistral:7b,gemma2:9b
```

```
LEADERBOARD — 3 model(s) on suites/default.toml
════════════════════════════════════════════════════════════════════
                     🏆  FINAL LEADERBOARD  🏆
════════════════════════════════════════════════════════════════════
  RANK  MODEL                             SCORE   PASS/TOTAL
────────────────────────────────────────────────────────────────────
🥇 #1  mistral:7b                        0.954      5/5
🥈 #2  llama3.1:8b                       0.920      4/5
🥉 #3  gemma2:9b                         0.810      4/5
────────────────────────────────────────────────────────────────────

  Best model: mistral:7b (score 0.954, 5/5 passed)
  ⚠ Score spread of 14.4% — models differ significantly on this suite.
```

Works with all standard flags — `--judge`, `--n-runs`, `--filter`, `--concurrency`:

```bash
# Safety audit across three models, 3 runs each for stable scores
crucible run --suite suites/safety/owasp_llm01_injection.toml \
  --models llama3.1:8b,mistral:7b,phi4:14b \
  --n-runs 3 \
  --judge llama3.1:8b
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

### General

| Suite | Path | Coverage |
|---|---|---|
| Default eval | `suites/default.toml` | Factual, JSON, instruction-following, basic safety |
| RAG faithfulness | `suites/rag/faithfulness.toml` | Groundedness, no hallucination, context relevance |
| RAG fallback | `suites/rag/fallback_chain.toml` | Typo robustness, graceful decline, semantic normalisation |

### OWASP LLM Top 10 — Full Coverage

54 probes across all 10 risk categories. Run any suite individually or the whole directory at once:

```bash
# Run a single risk category
crucible run --suite suites/safety/owasp_llm06_excessive_agency.toml

# Run the entire OWASP Top 10
crucible run --dir suites/safety --category safety
```

| # | Risk | Suite | Probes |
|---|---|---|---|
| LLM01 | Prompt Injection | `suites/safety/owasp_llm01_injection.toml` | Direct injection, role hijack, indirect via context, delimiter confusion, system prompt exfiltration |
| LLM02 | Sensitive Information Disclosure | `suites/safety/owasp_llm02_sensitive_disclosure.toml` | PII leakage from context, credential exposure, training data probing, cross-user isolation |
| LLM03 | Supply Chain | `suites/safety/owasp_llm03_supply_chain.toml` | Malicious plugin output, third-party authority spoof, poisoned knowledge base, provenance uncertainty |
| LLM04 | Data & Model Poisoning | `suites/safety/owasp_llm04_data_poisoning.toml` | Fact overwrite via context, authority injection, backdoor triggers, gradual belief manipulation, bias injection |
| LLM05 | Improper Output Handling | `suites/safety/owasp_llm05_output_handling.toml` | SQL injection in generated queries, XSS in HTML output, shell injection, template injection, path traversal |
| LLM06 | Excessive Agency | `suites/safety/owasp_llm06_excessive_agency.toml` | Destructive actions without confirmation, bulk email, financial transfers, self-modification, production actions |
| LLM07 | System Prompt Leakage | `suites/safety/owasp_llm07_system_prompt_leakage.toml` | Direct requests, base64/translation extraction, forbidden topic enumeration, roleplay extraction |
| LLM08 | Vector & Embedding Weaknesses | `suites/safety/owasp_llm08_vector_weaknesses.toml` | Adversarial chunk hijack, authority spoof in retrieved docs, cross-user leakage, conflicting/stale retrieval |
| LLM09 | Misinformation | `suites/safety/owasp_llm09_misinformation.toml` | Hallucinated citations, fabricated statistics, medical/legal/financial advice, fabricated quotes |
| LLM10 | Unbounded Consumption | `suites/safety/owasp_llm10_unbounded_consumption.toml` | Token amplification, recursive expansion, context stuffing, infinite loop traps, latency SLA enforcement |

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

- **v0.1** (now) — Core eval runner, mechanism detection, SQLite regression, full OWASP LLM Top 10
- **v0.2** — RAG IR metrics (Recall@k, MRR), long-context needle tests, latency as first-class metric
- **v0.3** — Agent multi-turn loop eval, LlamaGuard integration, cost tracking ($/run)
- **v0.4** — MMLU/HumanEval benchmark runners, statistical significance, JUnit XML output

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
