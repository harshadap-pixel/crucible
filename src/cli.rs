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
    Run(RunArgs),

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
}

// ── autodiscover ──────────────────────────────────────────────────────────────

#[derive(Args, Debug)]
pub struct AutodiscoverArgs {
    /// Directory to scan for AI code (eval runners, NL2SQL, RAG, etc.)
    #[arg(short, long)]
    pub dir: String,

    /// Run generated suites immediately after discovery
    #[arg(long, default_value_t = false)]
    pub run: bool,

    /// Save generated suite files to this directory (default: system temp dir)
    #[arg(long)]
    pub save: Option<String>,

    /// Model override for generated suites that need one (default: mock)
    #[arg(short, long)]
    pub model: Option<String>,

    /// Ollama base URL (used when --run is set)
    #[arg(long, default_value = "http://localhost:11434")]
    pub ollama_url: String,
}

// ── run ──────────────────────────────────────────────────────────────────────

#[derive(Args, Debug)]
pub struct RunArgs {
    // ── Source selection ──────────────────────────────────────────────────────
    /// Single suite file to run
    #[arg(short, long, default_value = "suites/default.toml")]
    pub suite: String,

    /// Run every .toml suite found recursively in this directory
    #[arg(long, conflicts_with = "suite")]
    pub dir: Option<String>,

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
    #[arg(long, default_value = "http://localhost:11434")]
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
    #[arg(long, default_value = "http://localhost:11434")]
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
