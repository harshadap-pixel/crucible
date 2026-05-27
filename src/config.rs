use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

// ── Top-level suite ───────────────────────────────────────────────────────────

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Suite {
    pub suite: SuiteMeta,
    #[serde(default)]
    pub tests: Vec<TestCase>,
}

/// Optional script provider — when present, tests are run via subprocess
/// instead of Ollama. cmd + args are called with the prompt on stdin.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ScriptConfig {
    /// Executable to run (e.g. "python3")
    pub cmd: String,
    /// Arguments passed before the prompt (e.g. path to adapter script)
    #[serde(default)]
    pub args: Vec<String>,
    /// Extra environment variables injected into the subprocess
    #[serde(default)]
    pub env: Vec<[String; 2]>,
}

/// Optional HTTP provider — when present, tests POST/GET a real endpoint
/// instead of calling a model. Used for full-stack API testing.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct HttpConfig {
    /// Base URL of the service (e.g. "http://localhost:3000")
    pub base_url: String,
    /// Headers sent with every request (e.g. Authorization)
    #[serde(default)]
    pub headers: std::collections::HashMap<String, String>,
    /// Request timeout in milliseconds
    #[serde(default = "default_http_timeout_ms")]
    pub timeout_ms: u64,
}

fn default_http_timeout_ms() -> u64 {
    10_000
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct SuiteMeta {
    pub name: String,
    pub model: String,
    #[serde(default = "default_judge")]
    pub judge: String,
    #[serde(default = "default_ollama_url")]
    pub ollama_url: String,
    #[serde(default = "default_concurrency")]
    pub concurrency: usize,
    #[serde(default = "default_n_runs")]
    pub n_runs: u32,
    /// Score drop threshold that triggers a regression flag (0.0–1.0)
    #[serde(default = "default_regression_threshold")]
    pub regression_threshold: f64,
    /// Category tag for filtering (e.g. "safety", "rag", "benchmark")
    #[serde(default)]
    pub category: Option<String>,
    /// When set, use subprocess provider instead of Ollama for model calls
    #[serde(default)]
    pub script: Option<ScriptConfig>,

    /// When set, use HTTP provider — tests call a live API endpoint
    #[serde(default)]
    pub http: Option<HttpConfig>,
}

fn default_judge() -> String {
    "llama3.1:8b".into()
}
fn default_ollama_url() -> String {
    "http://localhost:11434".into()
}
fn default_concurrency() -> usize {
    4
}
fn default_n_runs() -> u32 {
    1
}
fn default_regression_threshold() -> f64 {
    0.05
}

// ── Multi-turn conversation ───────────────────────────────────────────────────

/// A single turn in a multi-turn conversation test.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Turn {
    /// "user" or "assistant"
    pub role: String,
    pub content: String,
}

// ── Test case ─────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct TestCase {
    pub name: String,

    /// Final user prompt (or full prompt for single-turn tests).
    /// For multi-turn tests this is appended as the last user turn.
    #[serde(default)]
    pub prompt: String,

    /// Multi-turn conversation history (drives context-retention tests).
    /// The runner replays these turns before sending `prompt`.
    #[serde(default)]
    pub turns: Vec<Turn>,

    // ── HTTP provider fields (used when [suite.http] is configured) ──────────
    /// HTTP method: "GET" | "POST" | "PUT" | "DELETE"
    #[serde(default)]
    pub method: Option<String>,

    /// Path relative to [suite.http].base_url (e.g. "/api/ai/nl2sql")
    #[serde(default)]
    pub path: Option<String>,

    /// Request body sent as-is (usually JSON)
    #[serde(default)]
    pub body: Option<String>,

    /// Per-request headers that override suite-level headers
    #[serde(default)]
    pub headers: std::collections::HashMap<String, String>,

    // ── SSE / streaming ─────────────────────────────────────────────────────
    /// Treat HTTP response as a Server-Sent Events stream.
    /// Automatically enabled when Content-Type: text/event-stream is detected.
    /// Set explicitly to true to force SSE accumulation regardless of header.
    #[serde(default)]
    pub sse: bool,

    /// How long to collect SSE events before closing the stream (ms).
    /// Defaults to the suite-level http.timeout_ms when not set.
    #[serde(default)]
    pub sse_timeout_ms: Option<u64>,

    // ── Standard fields ──────────────────────────────────────────────────────
    /// Optional retrieved context chunks (for RAG tests)
    #[serde(default)]
    pub context: Vec<String>,

    /// Variable substitutions applied to prompt ({{key}} → value)
    #[serde(default)]
    pub vars: std::collections::HashMap<String, String>,

    /// Simulated tool outputs for agent tests
    #[serde(default)]
    pub tool_outputs: Vec<ToolOutput>,

    pub assert: Vec<Assertion>,

    /// Per-test override: how many samples to average
    #[serde(default)]
    pub n_runs: Option<u32>,

    /// Human-readable description shown in report
    #[serde(default)]
    pub description: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ToolOutput {
    pub tool: String,
    pub result: String,
}

// ── Assertions ────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Assertion {
    /// Output must contain this exact substring
    Contains {
        value: String,
        #[serde(default)]
        weight: f64,
    },
    /// Output must NOT contain this substring
    NotContains {
        value: String,
        #[serde(default)]
        weight: f64,
    },
    /// Output must match this regex
    Regex {
        value: String,
        #[serde(default)]
        weight: f64,
    },
    /// Output must match exactly
    ExactMatch {
        value: String,
        #[serde(default)]
        weight: f64,
    },
    /// Output must be valid JSON matching this schema
    JsonSchema {
        schema: String,
        #[serde(default)]
        weight: f64,
    },
    /// Semantic similarity to reference (via embeddings)
    Semantic {
        reference: String,
        #[serde(default = "default_sim_threshold")]
        threshold: f64,
        #[serde(default)]
        weight: f64,
    },
    /// LLM-as-judge rubric scoring
    LlmJudge {
        rubric: String,
        #[serde(default = "default_judge_threshold")]
        threshold: f64,
        #[serde(default)]
        weight: f64,
    },
    /// Model refused to answer (safety tests)
    RefusalCheck {
        #[serde(default)]
        weight: f64,
    },
    /// Specific tool was NOT called (agent safety)
    ToolNotCalled {
        tool: String,
        #[serde(default)]
        weight: f64,
    },

    // ── New assertion types ───────────────────────────────────────────────────
    /// Response must arrive within this many milliseconds.
    /// Fails (score=0) if latency_ms > ms.
    LatencyUnder {
        ms: u64,
        #[serde(default)]
        weight: f64,
    },

    /// Navigate a JSON path and assert on the value at that node.
    /// path uses dot-notation: "$.field", "$.nested.field", "$[0].field"
    JsonField {
        path: String,
        /// Value must equal this string exactly (after JSON serialization)
        #[serde(default)]
        equals: Option<String>,
        /// Value must contain this substring
        #[serde(default)]
        contains: Option<String>,
        #[serde(default)]
        weight: f64,
    },

    /// HTTP response must have this status code.
    /// Only evaluated when [suite.http] is configured.
    HttpStatus {
        code: u16,
        #[serde(default)]
        weight: f64,
    },

    /// Compare output against a golden file.
    /// First run (file absent): writes the output as the new golden — always passes.
    /// Subsequent runs: fails if output does not match the file exactly (after trim).
    /// Re-generate with: crucible run --update-snapshots
    Snapshot {
        /// Path to the golden file (relative to the suite file's directory,
        /// or absolute). Directories are created automatically.
        file: String,
        #[serde(default)]
        weight: f64,
    },
}

fn default_sim_threshold() -> f64 {
    0.75
}
fn default_judge_threshold() -> f64 {
    0.70
}

impl Assertion {
    pub fn weight(&self) -> f64 {
        let w = match self {
            Self::Contains { weight, .. } => *weight,
            Self::NotContains { weight, .. } => *weight,
            Self::Regex { weight, .. } => *weight,
            Self::ExactMatch { weight, .. } => *weight,
            Self::JsonSchema { weight, .. } => *weight,
            Self::Semantic { weight, .. } => *weight,
            Self::LlmJudge { weight, .. } => *weight,
            Self::RefusalCheck { weight, .. } => *weight,
            Self::ToolNotCalled { weight, .. } => *weight,
            Self::LatencyUnder { weight, .. } => *weight,
            Self::JsonField { weight, .. } => *weight,
            Self::HttpStatus { weight, .. } => *weight,
            Self::Snapshot { weight, .. } => *weight,
        };
        if w == 0.0 {
            1.0
        } else {
            w
        }
    }
}

// ── Loader ────────────────────────────────────────────────────────────────────

pub fn load(path: &str) -> Result<Suite> {
    let raw = std::fs::read_to_string(Path::new(path))
        .with_context(|| format!("Cannot read suite file: {path}"))?;
    let suite: Suite =
        toml::from_str(&raw).with_context(|| format!("Invalid TOML in suite: {path}"))?;
    Ok(suite)
}
