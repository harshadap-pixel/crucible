# Crucible — System Design

> Version 0.1.20 · LLM evaluation for quality, RAG, agents, safety, and mechanism detection — local and cloud

---

## 1. Problem Statement

Evaluating local LLMs is painful in 2025. The existing tools all share a common constraint: **they operate on Q&A pairs you provide, not on your code**. You bring the questions, they score the answers. None of them look at your codebase.

| Gap | Consequence |
|-----|-------------|
| Cloud-only eval tools | Latency, cost, data privacy concerns |
| No code scanning | Can't match eval strategy to actual pipeline implementation |
| Strategy-blind tests | Generic faithfulness/relevance tests miss RAG-specific failure modes |
| Complex setup | API keys, Python envs, config files just to run one test |
| No regression tracking | No baseline → no signal when a model update silently degrades output |

Crucible's answer: **one static binary, Ollama-native with full cloud provider support, and a built-in code scanner that detects what your AI pipeline actually does and generates tests targeting the precise failure modes of each detected strategy**.

---

## 2. Goals and Non-Goals

### Goals
- Run locally with Ollama or against any cloud provider with a single prefix change
- Detect AI code patterns via static analysis and generate matching eval suites
- Strategy-aware RAG evaluation (reranking, hybrid, HyDE, multi-query, chunking, etc.)
- Persistent run history, regression baselines, and diff-based comparison
- Dataset evaluation with MMLU preset, template expansion, and slice analysis
- Single binary install — `cargo install` or `crucible update`

### Non-Goals
- Real-time production monitoring
- Fine-tuning or training data generation
- UI / web dashboard (terminal-first by design)

---

## 3. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                  CLI (clap)                                 │
│  run │ detect │ baseline │ compare │ report │ status │ autodiscover │ update │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │
             ┌───────────────────────┼────────────────────┐
             │                       │                    │
             ▼                       ▼                    ▼
    ┌─────────────────┐   ┌──────────────────┐   ┌──────────────────┐
    │     Runner      │   │    Autodiscover   │   │  Detect Engine   │
    │  runner.rs      │   │  discover/        │   │  detect/         │
    │  leaderboard.rs │   │  scanner.rs       │   │  model_meta.rs   │
    │  dataset.rs     │   │  generator.rs     │   │  pipeline.rs     │
    └────────┬────────┘   └──────────────────┘   └──────────────────┘
             │
    ┌────────┴────────────────────────┐
    │                                 │
    ▼                                 ▼
┌──────────────┐            ┌──────────────────┐
│  Providers   │            │   Assertions     │
│  Ollama      │            │  deterministic   │
│  HTTP        │            │  semantic        │
│  Script      │            │  llm_judge       │
└──────────────┘            └──────────────────┘
             │                       │
             └───────────┬───────────┘
                         ▼
                ┌─────────────────┐
                │   Store (SQLite)│
                │  db.rs          │
                │  baseline.rs    │
                └────────┬────────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
     ┌──────────────┐    ┌──────────────────┐
     │   Report     │    │   Regression     │
     │  report.rs   │    │  regression.rs   │
     └──────────────┘    └──────────────────┘
```

---

## 4. Component Deep-Dive

### 4.1 CLI Layer (`src/cli.rs`)

Built with `clap` (derive macros). All subcommands are strongly typed:

```
crucible
├── run         RunArgs          — execute suites, datasets, leaderboards
├── detect      DetectArgs       — probe model architecture + pipeline
├── baseline    BaselineArgs     — manage regression baselines (set | show)
├── compare     CompareArgs      — diff two run IDs
├── report      ReportArgs       — run history
├── status                       — summary of suites + current baseline
├── models                       — list detected providers + configured API keys
├── autodiscover AutodiscoverArgs — scan codebase + generate suites
└── update                       — self-update from GitHub releases
```

`RunArgs` is heap-allocated (`Box<RunArgs>`) because it accumulates ~336 bytes of flags — large enough to trigger the `large_enum_variant` Clippy lint when stored inline in the `Command` enum.

### 4.2 Runner (`src/runner.rs`, `src/dataset.rs`, `src/leaderboard.rs`)

**Three execution modes**, selected by the presence of CLI flags:

| Mode | Trigger | What happens |
|------|---------|--------------|
| Standard | _(default)_ | Run `.toml` suite(s) against one model |
| Dataset | `--dataset` + `--template` | Load JSONL/CSV, expand template, run N test cases |
| Leaderboard | `--models a,b,c` | Run same suite against multiple models, print ranking table |

**Standard run flow:**

```
RunArgs
  └─ resolve_suites()         # CWD-first, then embedded suites dir
       └─ for each suite path:
            load_suite()      # parse TOML → Suite struct
            apply_cli_overrides()  # model, judge, concurrency, n_runs
            execute_suite()
              └─ Semaphore(concurrency)
                   └─ for each TestCase (parallel):
                        provider.generate(prompt)  → output + latency + TTFT + tokens
                        assertions::evaluate_all() → (weighted_score, Vec<AssertionResult>)
                        → TestResult
              └─ aggregate: avg_score, passed/total
              └─ store.save_run()
              └─ regression::compare_to_baseline() (if --compare)
              └─ report::print_results()
```

**Dataset mode** (`src/dataset.rs`):
- Detects MMLU CSV format (question/A/B/C/D/answer columns) and auto-normalises to `{question, options, correct}`
- Expands `{{field}}` placeholders in suite prompt/assertions against each dataset row
- Computes per-slice score histograms, TTFT percentiles, pass-rate distributions
- `--slice-by category` groups results by a named column for segment analysis

**Concurrency**: `tokio::sync::Semaphore` gates parallel test execution. Default: 4. Test cases within a suite run in parallel; suites themselves run sequentially to avoid resource exhaustion.

### 4.3 Providers (`src/providers/`)

All providers implement the `Provider` trait (`chat`, `chat_with_history`, `name`). A model is addressed via a URI-prefix string resolved by `ModelRef::resolve()`:

| Prefix | Provider | Auth | Transport |
|--------|----------|------|-----------|
| _(none)_ or `ollama:` | `OllamaClient` | none | HTTP streaming (Ollama) |
| `openai:` | `OpenAICompatProvider` | `OPENAI_API_KEY` | SSE streaming |
| `groq:` | `OpenAICompatProvider` | `GROQ_API_KEY` | SSE streaming |
| `together:` | `OpenAICompatProvider` | `TOGETHER_API_KEY` | SSE streaming |
| `mistral:` | `OpenAICompatProvider` | `MISTRAL_API_KEY` | SSE streaming |
| `openrouter:` | `OpenAICompatProvider` | `OPENROUTER_API_KEY` | SSE streaming |
| `anthropic:` | `AnthropicProvider` | `ANTHROPIC_API_KEY` | SSE streaming |
| `azure:` | `OpenAICompatProvider` (Azure mode) | `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT` | SSE streaming |
| `bedrock:` | `BedrockProvider` | `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` | HTTPS (non-streaming) |
| `[suite.http]` | `HttpProvider` | per-suite headers | HTTP POST |
| `[suite.script]` | `ScriptProvider` | n/a | subprocess stdin/stdout |

**Ollama streaming** captures TTFT (time to first token) by timing the gap between `POST /api/generate` and the first streamed chunk. Full latency is end-to-end wall time. Token counts come from Ollama's `eval_count` / `prompt_eval_count` fields in the final chunk.

**OpenAI-compatible providers** (`OpenAICompatProvider`) use SSE streaming (`data: {...}` lines, `[DONE]` sentinel) with `stream_options: { include_usage: true }` to capture token counts in the final chunk. TTFT is recorded on the first non-empty content delta.

**Azure OpenAI** has three quirks handled automatically:
- Auth: `api-key` header (not `Authorization: Bearer`)
- Newer models require `max_completion_tokens` instead of `max_tokens`
- o-series / reasoning models reject explicit `temperature` — omitted entirely. Detected via `is_reasoning_deployment()` heuristic (o1*, o3*, *codex*, gpt-5*, *-chat)
- Content-filter 400 responses (`content_filter` / `ResponsibleAIPolicyViolation`) are caught and converted to a synthetic `[CONTENT_FILTERED]` refusal, triggering `refusal_check` assertions correctly

**Anthropic** uses the Messages API SSE format: `message_start` (input tokens), `content_block_delta` (text chunks), `message_delta` (output tokens + stop reason).

**AWS Bedrock** (`BedrockProvider`) uses the Converse API — a unified non-streaming interface across all Bedrock model families (Claude, Llama, Mistral, Titan, Cohere). Request signing uses hand-rolled SigV4 (`AWS4-HMAC-SHA256`) via the `hmac` + `sha2` crates. Supports `AWS_SESSION_TOKEN` for IAM roles and SSO. TTFT equals total latency (non-streaming). Model IDs use full Bedrock format, e.g. `anthropic.claude-3-5-sonnet-20241022-v2:0`.

**Judge auto-coercion** (`coerce_judge()`): when the judge is still the default `llama3.1:8b` but the eval model uses a cloud prefix, the judge is automatically mirrored to the cheapest model on the same provider (e.g. `openai:gpt-4o-mini`, `anthropic:claude-3-haiku-20240307-v1:0`). This prevents failures when Ollama is not running.

**Provider detection** (`detect_available()`): scans env vars at startup and reports which providers are configured. `crucible models` displays the full status table. `auto_select_model()` picks the best available model in priority order: Ollama (local, free) → Groq → OpenAI → Anthropic → …

### 4.4 Assertions (`src/assertions/`)

Weighted scoring model. Every assertion carries an optional `weight` (default: `1.0`). Final test score = Σ(score × weight) / Σ(weight).

| Type | Module | Description |
|------|--------|-------------|
| `contains` | deterministic | Substring present |
| `not_contains` | deterministic | Substring absent |
| `regex` | deterministic | Regex match |
| `exact_match` | deterministic | Trimmed string equality |
| `json_schema` | deterministic | JSON Schema v7 validation |
| `json_field` | deterministic | JSONPath-like field equals/contains |
| `http_status` | deterministic | HTTP response code check |
| `latency_under` | deterministic | Wall-time budget (ms) |
| `ttft_under` | deterministic | First-token budget (ms) |
| `snapshot` | deterministic | Golden file diff; `--update-snapshots` to refresh |
| `semantic` | semantic | Cosine similarity via Ollama embeddings |
| `llm_judge` | llm_judge | LLM-as-judge with rubric + threshold (0–1) |
| `refusal_check` | deterministic | Heuristic: does output contain a refusal pattern? |
| `tool_not_called` | deterministic | Tool name absent from JSON output |

**LLM judge flow**: Sends a structured prompt to the judge model asking for a score 0–10 with brief reasoning. Parses the first integer from the response, divides by 10, applies threshold. Judge model defaults to `llama3.1:8b`; overridable per-suite or via `--judge`. When the eval model is a cloud provider, `coerce_judge()` automatically upgrades the judge to a matching cloud model so Ollama doesn't need to be running.

**Per-test system prompts**: `TestCase` supports an optional `system` field that overrides the suite-level system prompt for individual tests.

**Semantic similarity**: Embeds both the model output and the reference string using Ollama's `/api/embeddings` endpoint, computes cosine similarity.

### 4.5 Autodiscover Pipeline (`src/discover/`)

The most differentiated component. Scans a codebase statically, detects AI patterns, then generates targeted eval suites.

```
scan(dir)
  └─ walk_dir() — recursive, skip venvs/node_modules/build artifacts
       └─ for each file (*.py, *.ts, *.js, *.rs, *.go, *.yaml, *.json):
            detect_rag_profile()    → RagProfile (5 dimensions)
            detect_nl2sql()         → Option<signals>
            detect_ai_service()     → Option<signals>
            detect_eval_runner()    → Option<signals>
            detect_mcp_server()     → Option<signals>
            → Vec<Finding>

Phase 2: map findings → bundled suites
Phase 3: generate custom suites from RagProfile
Phase 4: (optional --run) execute all matched suites
```

**Skip list** (prevents venv/build noise):
```
node_modules, dist, .next, coverage, build, .turbo, vendor, target, .git
.env, env, venv, .venv, virtualenv
__pycache__, site-packages, dist-packages, .eggs, egg-info
.tox, .mypy_cache, .pytest_cache, .ruff_cache
```

**Finding kinds**:

| Kind | Signals looked for |
|------|--------------------|
| `RagPipeline` | Chunking, reranking, retrieval strategies, framework imports, embedding models |
| `NL2Sql` | SQL generation patterns, text2sql markers |
| `AiService` | LLM client init, provider imports (OpenAI, Anthropic, etc.) |
| `EvalRunner` | RAGAS, DeepEval, TruLens, Promptfoo imports |
| `McpServer` | MCP server/tool decorators, FastMCP |

#### 4.5.1 RAG Strategy Detection — `RagProfile`

The `RagProfile` struct captures five independent dimensions from static code analysis:

```rust
pub struct RagProfile {
    pub chunking:   Vec<String>,  // chunking strategy names
    pub reranking:  Vec<String>,  // reranker model/method names
    pub retrieval:  Vec<String>,  // retrieval strategies + vector stores
    pub frameworks: Vec<String>,  // orchestration frameworks
    pub embedding:  Vec<String>,  // embedding model providers
}
```

**Chunking signals** (`RecursiveCharacterTextSplitter`, `SemanticChunker`, `SentenceWindowNodeParser`, `TokenTextSplitter`, `CodeSplitter`, `late_chunking`, etc.)

**Reranking signals** (`CrossEncoderRanker`, `CohereRerank`, `BGEReranker`, `ColBERT`, `llm_rerank`, `VoyageRerank`)

**Retrieval signals**:
- Strategies: `hybrid`, `BM25`, `HyDE`, `MultiQueryRetriever`, `StepBackRetriever`, `ContextualCompressionRetriever`, `ParentDocumentRetriever`
- Vector stores: Pinecone, Weaviate, Qdrant, Chroma, Milvus, FAISS, PGVector, Elasticsearch, OpenSearch, Redis, Mongo Atlas, LanceDB

**Frameworks**: LangChain, LlamaIndex, Haystack, DSPy

**Embedding**: OpenAI, Cohere, SentenceTransformers, HuggingFace, Ollama, Voyage, ONNX

#### 4.5.2 Strategy-Aware Test Generation (`src/discover/generator.rs`)

Each detected strategy dimension triggers its own test block. Base tests (faithfulness + no-hallucination) are always emitted. Additional tests are conditional:

| Condition | Test generated | What it catches |
|-----------|---------------|-----------------|
| Always | `rag-faithfulness` | Hallucination beyond provided context |
| Always | `rag-no-hallucination` | Made-up entities |
| `chunking` detected | `rag-chunk-boundary` | Answer split across chunk boundary, both required |
| `reranking` detected | `rag-reranker-demotes-lexical-match` | Reranker choosing semantic over lexical relevance |
| `hybrid` retrieval | `rag-hybrid-keyword-precision` | BM25 lane: exact SKU/ID retrieval |
| `hybrid` retrieval | `rag-hybrid-semantic-fallback` | Vector lane: paraphrase with no keyword overlap |
| `hyde` retrieval | `rag-hyde-abstract-query` | Abstract question ↔ source with no shared vocabulary |
| `multi_query` | `rag-multi-query-disambiguation` | Ambiguous term ("Mercury") requires multi-angle retrieval |
| `late_chunking` | `rag-late-chunking-cross-sentence` | Cross-sentence context requiring late binding |
| `parent_doc` | `rag-parent-doc-full-statement` | Small chunk match, answer needs parent document scope |
| `contextual_compression` | `rag-contextual-compression-extraction` | 90% noise, extract 1 relevant sentence |

All `llm_judge` assertions are emitted as single-line TOML inline tables (TOML 1.0 spec compliance — multi-line inline tables are invalid).

### 4.6 Detect Engine (`src/detect/`)

Probes a running Ollama model to identify its architecture and pipeline mechanisms:

- **Model meta** (`model_meta.rs`): Parses `ollama show` JSON — family, parameter count, quantisation level, context length, embedding dimension
- **Pipeline** (`pipeline.rs`): KV cache latency probe (send identical prompt twice, measure latency delta → infers caching), static analysis of pipeline config files (YAML/TOML)

### 4.7 Storage (`src/store/`)

**Single SQLite file** at `~/Library/Application Support/crucible/crucible.db` (macOS) / `~/.local/share/crucible/crucible.db` (Linux). Zero system dependency — `rusqlite` bundles SQLite at compile time.

**Schema**:

```sql
-- One row per run
CREATE TABLE runs (
    id              TEXT PRIMARY KEY,
    suite_name      TEXT NOT NULL,
    model           TEXT NOT NULL,
    timestamp       TEXT NOT NULL,
    is_baseline     INTEGER NOT NULL DEFAULT 0,
    total_tests     INTEGER NOT NULL DEFAULT 0,
    passed_tests    INTEGER NOT NULL DEFAULT 0,
    avg_score       REAL NOT NULL DEFAULT 0,
    total_tokens    INTEGER NOT NULL DEFAULT 0
);

-- One row per test per run
CREATE TABLE test_results (
    run_id          TEXT NOT NULL REFERENCES runs(id),
    test_name       TEXT NOT NULL,
    score           REAL NOT NULL,
    passed          INTEGER NOT NULL,
    pass_rate       REAL NOT NULL DEFAULT 1.0,
    latency_ms      INTEGER NOT NULL DEFAULT 0,
    ttft_ms         INTEGER NOT NULL DEFAULT 0,
    input_tokens    INTEGER NOT NULL DEFAULT 0,
    output_tokens   INTEGER NOT NULL DEFAULT 0,
    reason          TEXT NOT NULL DEFAULT ''
);
```

**Baseline**: A `is_baseline = 1` flag on exactly one run per suite/model combination. `crucible baseline set` flips the flag; `crucible compare <run_a> <run_b>` diffs any two run IDs regardless of baseline status.

**Migration**: Schema additions use `ALTER TABLE ADD COLUMN` idiom at startup, ignoring "column already exists" errors — safe for existing databases.

### 4.8 Regression (`src/regression.rs`)

After every run (when `--compare` is set, default: true): loads the current baseline for the same suite+model, joins on `test_name`, reports score deltas. A test is flagged as a regression when:

```
score_new < score_baseline - suite.regression_threshold
```

Default threshold: `0.05` (5 percentage points). Configurable per-suite via `[suite].regression_threshold`.

### 4.9 Embedded Suites (`src/embedded.rs`)

`rust-embed` bakes all `suites/*.toml` files into the binary at compile time using the `Suites` struct. On first run, they're extracted to the platform data dir. Suite resolution is CWD-first, falling back to the extracted dir — this lets local overrides take precedence over bundled suites.

**Bundled suite library**:

```
suites/
├── default.toml                          # general quality + instruction following
├── benchmark/
│   ├── classify.toml                     # classification tasks
│   └── math.toml                         # mathematical reasoning
├── rag/
│   ├── faithfulness.toml                 # context adherence
│   └── fallback_chain.toml               # graceful degradation
└── safety/
    ├── owasp_llm01_injection.toml        # prompt injection
    ├── owasp_llm02_sensitive_disclosure.toml
    ├── owasp_llm03_supply_chain.toml
    ├── owasp_llm04_data_poisoning.toml
    ├── owasp_llm05_output_handling.toml
    ├── owasp_llm06_excessive_agency.toml
    ├── owasp_llm07_system_prompt_leakage.toml
    ├── owasp_llm08_vector_weaknesses.toml
    ├── owasp_llm09_misinformation.toml
    └── owasp_llm10_unbounded_consumption.toml
```

---

## 5. Data Flow

### 5.1 Standard Suite Run

```
CLI args (RunArgs)
  │
  ▼
resolve_suites()         # finds .toml files
  │
  ▼
Suite (parsed TOML)
  │  [suite.name, suite.model, suite.judge, [[tests]]]
  │
  ▼
execute_suite()
  │
  ├─ for each TestCase (parallel, semaphore-gated):
  │    │
  │    ├─ apply --var substitutions in prompt
  │    │
  │    ├─ OllamaClient::generate() ─────► POST /api/generate
  │    │    ◄── streaming JSON chunks ──
  │    │    → output: String
  │    │    → latency_ms: u64
  │    │    → ttft_ms: u64
  │    │    → input_tokens: u32
  │    │    → output_tokens: u32
  │    │
  │    └─ assertions::evaluate_all()
  │         → (weighted_score: f64, Vec<AssertionResult>)
  │
  ▼
SuiteOutcome { results: Vec<TestResult>, avg_score, passed, total }
  │
  ├─ store.save_run()  ──────────────────► SQLite (runs + test_results)
  │
  ├─ regression::compare_to_baseline()  ► load baseline → diff scores
  │
  └─ report::print_results()  ──────────► terminal / JSON / SARIF
```

### 5.2 Autodiscover Flow

```
scan(dir)
  │
  ├─ walk_dir() — skip venvs, build dirs, .git
  │    └─ for each source file:
  │         detect_rag_profile()    → RagProfile
  │         detect_nl2sql()         → signals
  │         detect_ai_service()     → signals
  │         detect_mcp_server()     → signals
  │
  ▼
Vec<Finding> { path, signals, kind: FindingKind }
  │
  ├─ bundled_suites_for(kind) → bundled suite paths
  │
  └─ generator::generate(findings, save_dir)
       └─ for RagPipeline findings:
            generate_rag_suite(finding, profile, out_dir)
              ├─ RAG_BASE_TESTS (always)
              ├─ rag_chunk_boundary_test()     if chunking detected
              ├─ rag_reranker_sensitivity_test() if reranking detected
              ├─ rag_hybrid_keyword_test()     if hybrid retrieval
              ├─ rag_hybrid_semantic_test()    if hybrid retrieval
              ├─ rag_hyde_test()               if hyde detected
              ├─ rag_multi_query_test()        if multi_query detected
              ├─ rag_late_chunking_test()      if late_chunking detected
              ├─ rag_parent_doc_test()         if parent_doc detected
              └─ rag_contextual_compression_test() if contextual_compression

  ▼
Vec<(path, description)> of generated .toml files
  │
  └─ (optional --run) runner::run() for each suite path
```

---

## 6. Suite TOML Format

```toml
[suite]
name        = "my-rag-eval"
model       = "llama3.1:8b"
judge       = "llama3.1:8b"          # for llm_judge assertions
concurrency = 4
n_runs      = 1                       # repeat each test N times
regression_threshold = 0.05
category    = "rag"                   # for --category filtering

# Optional: run against HTTP endpoint instead of Ollama
# [suite.http]
# base_url   = "http://localhost:3000"
# headers    = { Authorization = "Bearer ..." }
# timeout_ms = 10000

# Optional: run via subprocess
# [suite.script]
# cmd  = "python3"
# args = ["adapter.py"]

[[tests]]
name        = "faithfulness"
description = "Model should not add facts beyond context"
prompt      = "Context: {{context}}\nQuestion: {{question}}"
system      = "Answer only from the provided context."
n_runs      = 3                       # per-test override
assert = [
  { type = "not_contains",  value = "I don't know",       weight = 1.0 },
  { type = "llm_judge", rubric = "Answer uses only context facts. No hallucination.", threshold = 0.85, weight = 3.0 },
  { type = "latency_under", ms = 5000,                    weight = 0.5 },
]
```

---

## 7. Key Design Decisions

### 7.1 Single Static Binary

No Python environment, no Docker, no config files required. Install with `curl | sh`. Trades build-time complexity (Rust + cargo) for zero runtime dependencies. `rusqlite` is bundled; `rust-embed` bakes suites in.

### 7.2 Ollama-First with Full Cloud Support

The default and recommended path is local Ollama — zero cost, full privacy, no API keys. Cloud providers are first-class citizens via URI prefixes (`openai:`, `anthropic:`, `azure:`, `bedrock:`, `groq:`, `mistral:`, `together:`, `openrouter:`). This covers the full spectrum from air-gapped local dev to enterprise cloud deployments without changing the suite format.

### 7.3 Weighted Multi-Assertion Scoring

Rather than pass/fail per test, Crucible computes a weighted score per test. This allows:
- High-weight assertions for critical properties (semantic correctness)
- Low-weight assertions for soft constraints (latency budgets)
- Partial credit (a test that passes 3/5 assertions still records a non-zero score)
- Regression detection on score degradation rather than binary pass/fail flips

### 7.4 CWD-First Suite Resolution

Local suites in the project directory shadow embedded suites by name. This lets teams maintain project-specific suites alongside the bundled ones without reinstalling the binary.

### 7.5 Strategy-Aware Test Generation vs. Generic Tests

Generic RAG tests (faithfulness, relevance, groundedness) measure output quality but cannot tell you *why* the pipeline failed. Strategy-aware tests target the specific failure mode of each strategy:

- A reranker that's working correctly will *demote* lexically matching but semantically wrong results
- A hybrid retriever that's balanced correctly will handle both exact-ID queries (BM25) and paraphrase queries (vector) well
- HyDE works by generating a hypothetical document first — its test must have no keyword overlap between question and source

Detecting these strategies from code and emitting the right test is genuinely novel compared to all existing RAG evaluation tools.

### 7.6 TOML 1.0 Compliance for Generated Suites

TOML 1.0 spec: inline tables `{ key = value }` must be on a single line. Multi-line inline tables are a TOML 1.1 extension not yet supported by `toml` crate 0.8. All generated `llm_judge` assertions are emitted as single-line inline tables regardless of rubric length.

---

## 8. Release Pipeline

### Binary Distribution

GitHub Actions (`release.yml`) builds and publishes on every push to `main` (after a successful build — tagging only happens post-build to prevent dangling tags from failed runs):
- `x86_64-apple-darwin`
- `aarch64-apple-darwin` (Apple Silicon)

Build flags: `opt-level=3`, `lto=true`, `codegen-units=1`, `strip=true` → compact release binaries.

### Auto-Update

`crucible update` (`src/update.rs`):
1. Fetches `https://api.github.com/repos/harshadap-pixel/crucible/releases/latest`
2. Parses current semver vs. remote semver
3. Downloads matching platform binary to temp path
4. Atomically replaces `$0` (the running binary)

### CI (`ci.yml`)

Every push and PR runs:
```
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```

---

## 9. Competitive Landscape

| Tool | Code Scanning | Strategy Detection | Local + Cloud | Self-contained |
|------|:---:|:---:|:---:|:---:|
| **Crucible** | ✅ | ✅ | ✅ | ✅ |
| RAGAS | ❌ | ❌ | ❌ | ❌ |
| TruLens | ❌ | ❌ | ❌ | ❌ |
| DeepEval | ❌ | ❌ | ❌ | ❌ |
| Promptfoo | ❌ | ❌ | ✅ | ❌ |
| LlamaIndex Eval | ❌ | ❌ | ❌ | ❌ |
| Continuous Eval | ❌ | ❌ | ❌ | ❌ |

All existing tools operate on Q&A pairs you provide. Crucible is the only evaluator that scans your code to understand what your pipeline actually does, then generates tests targeting the precise failure modes of each detected strategy.

---

## 10. Extension Points

### Adding a New Provider

1. Implement the `Provider` trait in `src/providers/`:
   - `async fn chat(model, system, user, temperature) -> Result<CompletionResult>`
   - `async fn chat_with_history(model, history, final_user, temperature) -> Result<CompletionResult>`
   - `fn name() -> &'static str`
2. Add `pub mod your_provider;` to `src/providers/mod.rs`
3. Add a URI prefix branch in `ModelRef::resolve()` (e.g. `if let Some(m) = spec.strip_prefix("myprovider:")`)
4. Add env var detection in `detect_available()` and a mirror entry in `coerce_judge()`

### Adding a New Assertion Type

1. Add a variant to `config::Assertion`
2. Add the match arm in `assertions::evaluate_one()`
3. Implement scoring logic (returns `AssertionResult { kind, passed, score, reason, weight }`)

### Adding a New Scanner Pattern

1. Add a new `detect_*()` function in `scanner.rs`
2. Add the corresponding `FindingKind` variant
3. Map the kind to bundled suites in `discover::mod::bundled_suites_for()`
4. Add a `generate_*()` function in `generator.rs`

### Adding a New Bundled Suite

Drop a `.toml` file under `suites/` — `rust-embed` picks it up at next compile. Map the new path in `bundled_suites_for()` if it should be auto-matched by autodiscover.

---

## 11. Known Limitations

| Limitation | Current Behaviour | Potential Fix |
|------------|-----------------|---------------|
| Ollama must be running for semantic assertions | Embedding calls fail with connection error | Fallback to cloud embedding endpoint |
| No streaming progress for LLM judge calls | Looks frozen on slow judge models | Progress spinner per assertion |
| Generated suites have hardcoded prompts | No actual RAG context — tests assert model behaviour, not retrieval | Accept `--context` flag to inject real documents |
| Code scanner is text-pattern based | Can miss dynamic patterns, obfuscated imports | AST-based analysis (tree-sitter) |
| Single SQLite file | Not shareable across machines | Export/import command |
| No Windows or Linux binary | CI only builds macOS (x86_64 + aarch64) | Add `x86_64-unknown-linux-gnu` and `x86_64-pc-windows-msvc` targets |
| Bedrock Converse API is non-streaming | TTFT equals total latency | Use Bedrock streaming API |
