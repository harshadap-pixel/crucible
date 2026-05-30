pub mod generator;
pub mod scanner;

use anyhow::Result;
use colored::Colorize;

use crate::cli::AutodiscoverArgs;
use crate::cli::RunArgs;
use crate::embedded;
use crate::runner;
use scanner::{Finding, FindingKind};

/// Maps a finding to the bundled suite paths that are relevant for it.
/// These paths are relative to the suites dir (e.g. "rag/faithfulness.toml").
fn bundled_suites_for(kind: &FindingKind) -> Vec<&'static str> {
    match kind {
        FindingKind::RagPipeline { .. } => vec!["rag/faithfulness.toml", "rag/fallback_chain.toml"],
        FindingKind::AiService { .. } | FindingKind::EvalRunner { .. } => {
            vec!["default.toml", "safety/owasp_llm01_injection.toml"]
        }
        FindingKind::McpServer { .. } => vec!["default.toml", "safety/owasp_llm01_injection.toml"],
        FindingKind::NL2Sql { .. } => vec!["safety/owasp_llm01_injection.toml"],
    }
}

/// Entry point for `crucible autodiscover --dir <path>`
pub async fn run(args: AutodiscoverArgs) -> Result<()> {
    println!("\n{} {}", "AUTODISCOVER".bold().cyan(), args.dir.bold());
    println!("{}", "─".repeat(62).dimmed());
    println!("  Scanning for AI code patterns...\n");

    // ── Phase 1: Scan ─────────────────────────────────────────────
    let findings = scanner::scan(&args.dir)?;

    if findings.is_empty() {
        println!(
            "  {} No AI code patterns detected in {}",
            "○".dimmed(),
            args.dir.dimmed()
        );
        println!("  Try a directory that contains eval runners, LLM service files,");
        println!("  or RAG pipeline code (TypeScript or Python).");
        return Ok(());
    }

    // ── Phase 2: Print discovery report ──────────────────────────
    println!("  {} finding(s):\n", findings.len());
    for f in &findings {
        println!("  {} {}", "▸".cyan(), f.description().bold());
        println!("    {}", f.path.dimmed());
        println!("    signals: {}", f.signals.join(", ").dimmed());
        println!();
    }

    // ── Phase 3: Map findings → bundled suites ────────────────────
    let mut bundled: Vec<String> = Vec::new();
    for f in &findings {
        for suite in bundled_suites_for(&f.kind) {
            let resolved = embedded::resolve_suite_path(suite);
            if !bundled.contains(&resolved) {
                bundled.push(resolved);
            }
        }
    }

    if !bundled.is_empty() {
        println!("{}", "─".repeat(62).dimmed());
        println!("  {} bundled suite(s) matched:\n", bundled.len());
        for path in &bundled {
            // Print just the relative name, not the full installed path
            let display = std::path::Path::new(path)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or(path);
            println!("  {} {}", "→".green(), display.bold());
            println!("    {}", path.dimmed());
            println!();
        }
    }

    // ── Phase 4: Generate custom suites from code patterns ────────
    let save_dir = args.save.as_deref();
    let generated = generator::generate(&findings, save_dir)?;

    if !generated.is_empty() {
        println!("{}", "─".repeat(62).dimmed());
        println!(
            "  {} custom suite(s) generated from code:\n",
            generated.len()
        );
        for (path, desc) in &generated {
            println!("  {} {}", "→".green(), desc.bold());
            println!("    {}", path.dimmed());
            println!();
        }
    }

    if bundled.is_empty() && generated.is_empty() {
        println!(
            "  {} No runnable suites could be matched or generated.",
            "⚠".yellow()
        );
        return Ok(());
    }

    // ── Phase 5: Run if requested ─────────────────────────────────
    if args.run {
        println!("{}", "─".repeat(62).dimmed());
        println!("  {} Running suites...\n", "RUN".bold().cyan());

        // Run bundled suites first
        for suite_path in bundled.iter().chain(generated.iter().map(|(p, _)| p)) {
            let run_args = RunArgs {
                suite: suite_path.clone(),
                dir: None,
                category: None,
                filter: None,
                vars: vec![],
                model: args.model.clone(),
                judge: args.judge.clone(),
                models: vec![],
                ollama_url: args.ollama_url.clone(),
                concurrency: 4,
                n_runs: 1,
                fail_fast: false,
                retry: 0,
                output: "terminal".to_string(),
                compare: false,
                baseline: false,
                update_snapshots: false,
            };
            if let Err(e) = runner::run(run_args).await {
                eprintln!("  {} Suite failed: {e}", "✗".red());
            }
        }
    } else {
        println!(
            "  Tip: re-run with {} to execute all matched suites",
            "--run".yellow()
        );
        if generated.is_empty() {
            println!(
                "  Tip: re-run with {} to persist generated files",
                "--save ./suites/discovered".yellow()
            );
        }
    }

    Ok(())
}
