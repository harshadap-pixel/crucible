pub mod generator;
pub mod scanner;

use anyhow::Result;
use colored::Colorize;

use crate::cli::AutodiscoverArgs;
use crate::cli::RunArgs;
use crate::runner;

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
    println!("  {} findings:\n", findings.len());
    for f in &findings {
        println!("  {} {}", "▸".cyan(), f.description().bold());
        println!("    {}", f.path.dimmed());
        println!("    signals: {}", f.signals.join(", ").dimmed());
        println!();
    }

    // ── Phase 3: Generate suites ──────────────────────────────────
    let save_dir = args.save.as_deref();
    let suites = generator::generate(&findings, save_dir)?;

    if suites.is_empty() {
        println!("  {} No runnable suites could be generated.", "⚠".yellow());
        return Ok(());
    }

    println!("{}", "─".repeat(62).dimmed());
    println!("  {} suite(s) generated:\n", suites.len());
    for (path, desc) in &suites {
        println!("  {} {}", "→".green(), desc.bold());
        println!("    {}", path.dimmed());
        println!();
    }

    // ── Phase 4: Run if requested ─────────────────────────────────
    if args.run {
        println!("{}", "─".repeat(62).dimmed());
        println!("  {} Running generated suites...\n", "RUN".bold().cyan());

        for (suite_path, _) in &suites {
            let run_args = RunArgs {
                suite: suite_path.clone(),
                dir: None,
                category: None,
                filter: None,
                vars: vec![],
                model: args.model.clone(),
                judge: None,
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
            "  Tip: re-run with {} to execute all generated suites",
            "--run".yellow()
        );
        println!(
            "  Tip: re-run with {} to persist generated files",
            "--save ./suites/discovered".yellow()
        );
    }

    Ok(())
}
