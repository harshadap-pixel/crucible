# 🔥 Crucible

> Local-first LLM evaluation — quality, RAG, agents, safety, and mechanism detection in one binary.

**No Python. No Node. No cloud account. Just [Ollama](https://ollama.com).**

Every other eval tool tests what your model **outputs**.  
Crucible also tests what your pipeline **is**.

```
crucible run --model llama3:latest --judge llama3:latest

Run #a1b2c3d4  2026-05-30 14:23  model: llama3:latest
────────────────────────────────────────────────────────────────────────────
TEST                          SCORE   STATUS    LATENCY   TTFT    DELTA   FLAG
basic_factual                 1.000   ✅ PASS    312ms     48ms    +0.00
json_output                   1.000   ✅ PASS    289ms     61ms    +0.00
no_hallucination_on_unknown   0.920   ✅ PASS    401ms     74ms    -0.02
instruction_following         0.850   ✅ PASS    356ms     55ms    +0.01
refusal_jailbreak_basic       1.000   ✅ PASS    198ms     39ms    +0.00
────────────────────────────────────────────────────────────────────────────
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
| **TTFT measurement** | ❌ | ❌ | ❌ | ✅ |
| **Model leaderboard** | ❌ | ❌ | ❌ | ✅ |
| **Autodiscovery** | ❌ | ❌ | ❌ | ✅ |
| **Self-update** | ❌ | ❌ | ❌ | ✅ |
| Regression tracking | 🟡 | ❌ | ❌ | ✅ |
| Embedded SQLite | ✅ | ❌ | ❌ | ✅ |

---

## Install

### macOS (Apple Silicon)

```bash
gh release download --repo harshadap-pixel/crucible \
  --pattern "crucible-aarch64-macos" \
  --output ~/.local/bin/crucible --clobber \
  && chmod +x ~/.local/bin/crucible
```

### macOS (Intel)

```bash
gh release download --repo harshadap-pixel/crucible \
  --pattern "crucible-x86_64-macos" \
  --output ~/.local/bin/crucible --clobber \
  && chmod +x ~/.local/bin/crucible
```

Requires the [GitHub CLI](https://cli.github.com). Make sure `~/.local/bin` is on your `$PATH`:

```bash
export PATH="$HOME/.local/bin:$PATH"   # add to ~/.zshrc or ~/.bashrc
```

### From source

```bash
cargo install --git https://github.com/harshadap-pixel/crucible
```

### Updating

Once installed, update to the latest release with a single command — no `gh` needed:

```bash
crucible update
```

Crucible detects your OS and architecture, downloads the right binary, and replaces itself atomically.

---

## Quick Start

```bash
# 1. Start Ollama with any model
ollama serve
ollama pull llama3:latest

# 2. Run the default suite from anywhere — no project checkout needed
crucible run --model llama3:latest --judge llama3:latest

# 3. Set this run as your regression baseline
crucible baseline set

# 4. Make a change (swap model, tweak prompt, update RAG config...)
# 5. Run again — Crucible flags regressions automatically
crucible run --model llama3:latest --judge llama3:latest --compare
```

Suites are embedded in the binary and extracted automatically on first run. You never need to clone the repo.

---

## Commands

```
crucible run           [--suite <path>] [--model <name>] [--judge <name>]
crucible run           --models model1,model2,model3    # leaderboard mode
crucible run           --dir <category>                 # run all suites in a dir
crucible autodiscover  --dir <codebase> --model <name> --judge <name> [--run]
crucible detect        --model <name>   [--pipeline-config <path>]
crucible baseline      set | show
crucible compare       <run-id-a> <run-id-b>
crucible report        [--last N] [--suite <name>]
crucible status
crucible update
```

---

## Autodiscovery

Point Crucible at any codebase — it detects what AI patterns are in use and automatically runs the matching built-in suites:

```bash
crucible autodiscover --dir ~/my-project \
  --model llama3:latest --judge llama3:latest --run
```

```
AUTODISCOVER ~/my-project
──────────────────────────────────────────────────────────────
  Scanning for AI code patterns...

  2 finding(s):

  ▸ RAG pipeline — ONNX embeddings + USearch HNSW
    src/rag/pipeline.ts
    signals: onnxruntime, usearch, top_k

  ▸ AI service — Anthropic Claude
    src/services/llm.ts
    signals: anthropic

──────────────────────────────────────────────────────────────
  3 bundled suite(s) matched:

  → faithfulness.toml
  → fallback_chain.toml
  → owasp_llm01_injection.toml
```

| Code pattern detected | Suites automatically run |
|---|---|
| RAG pipeline | `rag/faithfulness.toml`, `rag/fallback_chain.toml` |
| AI service / eval runner | `default.toml`, `safety/owasp_llm01_injection.toml` |
| MCP server | `default.toml`, `safety/owasp_llm01_injection.toml` |
| NL2SQL validator | `safety/owasp_llm01_injection.toml` |

---

## Model Leaderboard

Compare any number of models on the same suite in one command:

```bash
crucible run --models llama3:latest,qwen2.5-coder:7b,mistral:7b \
  --judge llama3:latest
```

```
════════════════════════════════════════════════════════════════════
                     🏆  FINAL LEADERBOARD  🏆
════════════════════════════════════════════════════════════════════
  RANK  MODEL                             SCORE   PASS/TOTAL
────────────────────────────────────────────────────────────────────
🥇 #1  mistral:7b                        0.954      5/5
🥈 #2  llama3:latest                     0.640      2/5
🥉 #3  qwen2.5-coder:7b                  0.450      1/5
────────────────────────────────────────────────────────────────────

  Best model: mistral:7b (score 0.954, 5/5 passed)
  ⚠ Score spread of 50.4% — models differ significantly on this suite.
```

Works with all standard flags — `--suite`, `--judge`, `--n-runs`, `--filter`:

```bash
# Safety audit across three models, 3 runs each
crucible run --suite safety/owasp_llm01_injection.toml \
  --models llama3:latest,qwen2.5-coder:7b \
  --n-runs 3 --judge llama3:latest
```

---

## Built-in Suites

All suites are embedded in the binary — no file paths to memorise. Use short names:

```bash
crucible run --suite rag/faithfulness.toml --model llama3:latest --judge llama3:latest
crucible run --suite safety/owasp_llm01_injection.toml --model llama3:latest --judge llama3:latest
crucible run --dir benchmark --model llama3:latest --judge llama3:latest
```

### General

| Suite | Path | Coverage |
|---|---|---|
| Default | `default.toml` | Factual, JSON, instruction-following, basic safety |
| RAG faithfulness | `rag/faithfulness.toml` | Groundedness, no hallucination, context relevance |
| RAG fallback | `rag/fallback_chain.toml` | Typo robustness, graceful decline, semantic normalisation |

### Benchmark

| Suite | Path | Coverage |
|---|---|---|
| Math reasoning | `benchmark/math.toml` | Arithmetic, multi-step reasoning |
| Classification | `benchmark/classify.toml` | Sentiment classification |
| Question answering | `benchmark/qa.toml` | Factual QA |
| Structured extraction | `benchmark/extraction.toml` | JSON extraction from text |

### OWASP LLM Top 10 — Full Coverage

54 probes across all 10 risk categories:

```bash
# Single risk category
crucible run --suite safety/owasp_llm01_injection.toml --model llama3:latest --judge llama3:latest

# Entire OWASP Top 10
crucible run --dir safety --model llama3:latest --judge llama3:latest
```

| # | Risk | Probes |
|---|---|---|
| LLM01 | Prompt Injection | Direct injection, role hijack, indirect via context, delimiter confusion, system prompt exfiltration |
| LLM02 | Sensitive Information Disclosure | PII leakage, credential exposure, training data probing, cross-user isolation |
| LLM03 | Supply Chain | Malicious plugin output, third-party authority spoof, poisoned knowledge base |
| LLM04 | Data & Model Poisoning | Fact overwrite via context, authority injection, backdoor triggers, bias injection |
| LLM05 | Improper Output Handling | SQL injection in generated queries, XSS, shell injection, path traversal |
| LLM06 | Excessive Agency | Destructive actions without confirmation, bulk operations, financial transfers |
| LLM07 | System Prompt Leakage | Direct requests, base64/translation extraction, roleplay extraction |
| LLM08 | Vector & Embedding Weaknesses | Adversarial chunk hijack, cross-user leakage, conflicting retrieval |
| LLM09 | Misinformation | Hallucinated citations, fabricated statistics, medical/legal/financial advice |
| LLM10 | Unbounded Consumption | Token amplification, recursive expansion, context stuffing, latency SLA |

---

## TTFT (Time To First Token)

Every Ollama call uses streaming internally — Crucible timestamps the first token automatically. No configuration needed.

```
TEST                SCORE   STATUS    LATENCY   TTFT    DELTA
basic_factual       1.000   ✅ PASS    312ms     48ms    +0.00
json_output         1.000   ✅ PASS    289ms     61ms    +0.00
```

Assert on TTFT directly in any suite:

```toml
[[tests.assert]]
type = "ttft_under"
ms   = 300   # first token must arrive within 300ms
```

---

## Mechanism Detection

```bash
crucible detect --model mixtral:8x7b

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

⚠ MoE model detected — recommend --n-runs 5+ for stable scores
```

---

## Test Suite Format (TOML)

```toml
[suite]
name                 = "My RAG Pipeline"
model                = ""          # override with --model flag
judge                = ""          # override with --judge flag
concurrency          = 4
regression_threshold = 0.05        # flag if score drops >5%

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
| `regex` | Output must match this pattern (supports `(?i)` for case-insensitive) |
| `exact_match` | Output must equal this string exactly |
| `json_schema` | Output must be valid JSON matching this schema |
| `semantic` | Cosine similarity ≥ threshold (via Ollama embeddings) |
| `llm_judge` | LLM grades output against a rubric (0.0–1.0) |
| `refusal_check` | Output must look like a refusal (safety tests) |
| `latency_under` | End-to-end response time must be ≤ N ms |
| `ttft_under` | First token must arrive within N ms (Ollama only) |

---

## Regression Workflow

```bash
# Pin today's results as the baseline
crucible run --model llama3:latest --judge llama3:latest --baseline

# After changing your model or prompt, compare automatically
crucible run --model llama3:latest --judge llama3:latest --compare

# Explicitly compare two run IDs
crucible compare a1b2c3d4 e5f6g7h8

# View full history
crucible report
```

---

## JSON Output

Pipe to `jq` for scripting and CI integration:

```bash
# All results
crucible run --model llama3:latest --output json | jq '.tests[] | {test: .name, output: .output, score: .score}'

# Only failures
crucible run --model llama3:latest --output json | jq '.tests[] | select(.passed == false)'
```

SARIF output for GitHub Code Scanning:

```bash
crucible run --model llama3:latest --output sarif > results.sarif
```

---

## Roadmap

- **v0.1** (now) — Core eval runner, mechanism detection, SQLite regression, full OWASP LLM Top 10, model leaderboard, TTFT, autodiscovery, self-update
- **v0.2** — RAG IR metrics (Recall@k, MRR), long-context needle tests
- **v0.3** — Agent multi-turn loop eval, LlamaGuard integration, cost tracking ($/run)
- **v0.4** — MMLU/HumanEval benchmark runners, statistical significance, JUnit XML output

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
