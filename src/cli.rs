use clap::{Args, Parser, Subcommand};

#[derive(Parser)]
#[command(
    name    = "crucible",
    version,
    about   = "Local-first LLM evaluation — quality, RAG, agents, safety, and mechanism detection",
    long_about = None,
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// Run a test suite against a model
    Run(Box<RunArgs>),

    /// Detect model architecture and pipeline mechanisms
    Detect(DetectArgs),

    /// Manage regression baselines
    Baseline(BaselineArgs),

    /// Compare two runs for regressions
    Compare(CompareArgs),

    /// Show run history
    Report(ReportArgs),

    /// Show suite and baseline status
    Status,

    /// Scan a codebase for AI patterns and auto-generate + run eval suites
    Autodiscover(AutodiscoverArgs),

    /// List available local models and configured cloud providers
    Models(ModelsArgs),

    /// Update crucible to the latest release
    Update,

    /// Validate judge performance across multiple models
    ValidateJudge(ValidateJudgeArgs),
}

// ── models ────────────────────────────────────────────────────────────────────

#[derive(Args, Debug)]
pub struct ModelsArgs {
    /// Ollama base URL to probe for local models
    #[arg(long, default_value = "http://localhost:11434", env = "OLLAMA_HOST")]
    pub ollama_url: String,
}

// ── autodiscover ──────────────────────────────────────────────────────────────

#[derive(Args, Debug)]
pub struct AutodiscoverArgs {
    /// Directory to scan for AI code (eval runners, NL2SQL, RAG, etc.)
    /// Defaults to the current working directory.
    #[arg(short, long, default_value = ".")]
    pub dir: String,

    /// Run generated suites immediately after discovery
    #[arg(long, default_value_t = false)]
    pub run: bool,

    /// Save generated suite files to this directory (default: system temp dir)
    #[arg(long)]
    pub save: Option<String>,

    /// Model to evaluate (used when --run is set)
    #[arg(short, long)]
    pub model: Option<String>,

    /// Judge model for LLM-as-judge assertions (used when --run is set)
    #[arg(short, long)]
    pub judge: Option<String>,

    /// Ollama base URL (used when --run is set)
    #[arg(long, default_value = "http://localhost:11434", env = "OLLAMA_HOST")]
    pub ollama_url: String,
}

// ── run ──────────────────────────────────────────────────────────────────────

#[derive(Args, Debug)]
pub struct RunArgs {
    // ── Source selection ──────────────────────────────────────────────────────
    /// Single suite file to run
    #[arg(short, long, default_value = "default.toml")]
    pub suite: String,

    /// Run every .toml suite found recursively in this directory
    #[arg(long, conflicts_with = "suite")]
    pub dir: Option<String>,

    // ── Dataset mode ──────────────────────────────────────────────────────────
    /// Dataset file to evaluate (JSONL or CSV). Use with --template.
    #[arg(long)]
    pub dataset: Option<String>,

    /// Suite TOML used as a template when --dataset is set.
    /// Use {{field}} in prompt/assert to reference dataset columns.
    #[arg(long)]
    pub template: Option<String>,

    /// Group dataset results by this column (e.g. --slice-by category)
    #[arg(long)]
    pub slice_by: Option<String>,

    // ── Filtering ─────────────────────────────────────────────────────────────
    /// Only run suites whose [suite].category matches (use with --dir)
    #[arg(long)]
    pub category: Option<String>,

    /// Only run tests whose name contains this substring
    #[arg(long)]
    pub filter: Option<String>,

    // ── Runtime values ────────────────────────────────────────────────────────
    /// Inject a variable into all test prompts: --var KEY=value  (repeatable)
    #[arg(long = "var", value_name = "KEY=VALUE", action = clap::ArgAction::Append)]
    pub vars: Vec<String>,

    // ── Model / provider ──────────────────────────────────────────────────────
    /// Ollama model to evaluate (overrides suite setting)
    #[arg(short, long)]
    pub model: Option<String>,

    /// Judge model for LLM-as-judge assertions (overrides suite setting)
    #[arg(short, long)]
    pub judge: Option<String>,

    /// Compare multiple models side-by-side (leaderboard mode): --models a,b,c
    #[arg(long, value_delimiter = ',')]
    pub models: Vec<String>,

    /// Ollama base URL
    #[arg(long, default_value = "http://localhost:11434", env = "OLLAMA_HOST")]
    pub ollama_url: String,

    // ── Execution behaviour ───────────────────────────────────────────────────
    /// Max parallel test cases
    #[arg(long, default_value_t = 4)]
    pub concurrency: usize,

    /// Suite-level sample count; per-test n_runs takes priority when set
    #[arg(short, long, default_value_t = 1)]
    pub n_runs: u32,

    /// Stop the run after the first failing test
    #[arg(long, default_value_t = false)]
    pub fail_fast: bool,

    /// Retry a test N times on transport error before recording as failed
    #[arg(long, default_value_t = 0)]
    pub retry: u32,

    // ── Output ────────────────────────────────────────────────────────────────
    /// Output format: terminal (default) | json | sarif
    #[arg(long, default_value = "terminal")]
    pub output: String,

    // ── Post-run actions ──────────────────────────────────────────────────────
    /// Compare against current baseline after run
    #[arg(long, default_value_t = true)]
    pub compare: bool,

    /// Mark this run as the new regression baseline when it completes
    #[arg(long, default_value_t = false)]
    pub baseline: bool,

    /// Overwrite all snapshot golden files with current output instead of comparing
    #[arg(long, default_value_t = false)]
    pub update_snapshots: bool,
}

// ── detect ───────────────────────────────────────────────────────────────────

#[derive(Args, Debug)]
pub struct DetectArgs {
    /// Model to inspect
    #[arg(short, long)]
    pub model: String,

    /// Ollama base URL
    #[arg(long, default_value = "http://localhost:11434", env = "OLLAMA_HOST")]
    pub ollama_url: String,

    /// Path to pipeline config file (YAML/TOML) for static analysis
    #[arg(short, long)]
    pub pipeline_config: Option<String>,

    /// Run KV cache latency probe
    #[arg(long, default_value_t = true)]
    pub probe_cache: bool,
}

// ── baseline ─────────────────────────────────────────────────────────────────

#[derive(Args, Debug)]
pub struct BaselineArgs {
    #[command(subcommand)]
    pub action: BaselineAction,
}

#[derive(Subcommand, Debug)]
pub enum BaselineAction {
    /// Mark the most recent run as the regression baseline
    Set {
        /// Specific run ID to mark as baseline (defaults to last run)
        #[arg(long)]
        run_id: Option<String>,
    },
    /// Show current baseline info
    Show,
}

// ── compare ──────────────────────────────────────────────────────────────────

#[derive(Args, Debug)]
pub struct CompareArgs {
    /// First run ID (baseline)
    pub run_a: String,

    /// Second run ID (current)
    pub run_b: String,
}

// ── report ───────────────────────────────────────────────────────────────────

#[derive(Args, Debug)]
pub struct ReportArgs {
    /// Show last N runs
    #[arg(short, long, default_value_t = 10)]
    pub last: usize,

    /// Filter by suite name
    #[arg(short, long)]
    pub suite: Option<String>,
}

// ── validate-judge ───────────────────────────────────────────────────────────

#[derive(Args, Debug)]
pub struct ValidateJudgeArgs {
    /// Suite file to validate (default: all safety suites)
    #[arg(short, long, default_value = "safety")]
    pub suite: String,

    /// Model to run tests with (e.g., llama3.1:8b, qwen2.5-coder:7b, openai:gpt-4o)
    #[arg(short, long)]
    pub model: Option<String>,

    /// Judge models to compare (comma-separated)
    /// Defaults: anthropic:claude-fable-5,anthropic:claude-3-5-haiku-latest,groq:llama-3.1-8b-instant
    #[arg(long)]
    pub judges: Option<String>,

    /// Show divergent cases in detail
    #[arg(long, default_value_t = false)]
    pub show_divergent: bool,

    /// Fallback strategy: severity | skip | average | primary+fallback
    #[arg(long, default_value = "severity")]
    pub fallback_strategy: String,

    /// Fallback score for auth errors (401) when using severity strategy (0.0-1.0)
    #[arg(long, default_value = "0.0")]
    pub fallback_auth_score: f64,

    /// Fallback score for timeouts when using severity strategy (0.0-1.0)
    #[arg(long, default_value = "0.5")]
    pub fallback_timeout_score: f64,

    /// Fallback score for other errors when using severity strategy (0.0-1.0)
    #[arg(long, default_value = "0.5")]
    pub fallback_error_score: f64,

    /// Secondary judges to try if primary fails: judge1,judge2 (for primary+fallback strategy)
    #[arg(long)]
    pub fallback_judges: Option<String>,

    /// Skip test entirely if all judges fail (instead of using fallback scores)
    #[arg(long, default_value_t = false)]
    pub skip_on_all_fail: bool,

    /// Ollama base URL to probe for local models
    #[arg(long, default_value = "http://localhost:11434", env = "OLLAMA_HOST")]
    pub ollama_url: String,
}
