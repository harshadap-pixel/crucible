use anyhow::Result;
use colored::Colorize;
use std::collections::HashMap;
use std::sync::Arc;

use crate::cli::RunArgs;
use crate::providers::OllamaClient;
use crate::runner::{execute_suite, SuiteOutcome};

/// Entry point for leaderboard mode (`--models a,b,c`).
///
/// Runs the same suite against every model in `args.models`, then prints a
/// ranked table sorted by average score (highest first).
pub async fn run(
    args: RunArgs,
    suite_paths: Vec<String>,
    cli_vars: HashMap<String, String>,
) -> Result<()> {
    if suite_paths.len() > 1 {
        anyhow::bail!(
            "Leaderboard mode (--models) supports a single suite. \
             Use --suite instead of --dir."
        );
    }
    let suite_path = &suite_paths[0];

    println!(
        "\n{} — {} model(s) on {}",
        "LEADERBOARD".bold().cyan(),
        args.models.len(),
        suite_path.yellow(),
    );
    println!("{}", "═".repeat(68).dimmed());

    let embed_client = Arc::new(OllamaClient::new(&args.ollama_url));
    let mut outcomes: Vec<SuiteOutcome> = Vec::new();

    for model in &args.models {
        let outcome =
            execute_suite(suite_path, Some(model), &args, &cli_vars, &embed_client).await?;
        outcomes.push(outcome);
    }

    // Sort: highest avg_score first; tie-break: most tests passed
    outcomes.sort_by(|a, b| {
        b.avg_score
            .partial_cmp(&a.avg_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(b.passed.cmp(&a.passed))
    });

    // ── Print leaderboard table ───────────────────────────────────────────────
    println!("\n{}", "═".repeat(68).bold());
    println!("{:^68}", "🏆  FINAL LEADERBOARD  🏆".bold());
    println!("{}", "═".repeat(68).bold());

    println!(
        "{}",
        format!(
            "  {:<4}  {:<32}  {:>7}  {:>5}/{}",
            "RANK", "MODEL", "SCORE", "PASS", "TOTAL"
        )
        .dimmed()
    );
    println!("{}", "─".repeat(68).dimmed());

    let medals = ["🥇", "🥈", "🥉"];

    for (i, o) in outcomes.iter().enumerate() {
        let medal = medals.get(i).copied().unwrap_or("  ");
        let rank = format!("#{}", i + 1);
        let score_s = format!("{:.3}", o.avg_score);
        let score_colored = if o.avg_score >= 0.9 {
            score_s.green().to_string()
        } else if o.avg_score >= 0.7 {
            score_s.yellow().to_string()
        } else {
            score_s.red().to_string()
        };

        println!(
            "  {} {:<3}  {:<32}  {}  {:>5}/{}",
            medal, rank, o.model, score_colored, o.passed, o.total,
        );
    }

    println!("{}", "─".repeat(68).dimmed());

    // Winner callout
    if let Some(winner) = outcomes.first() {
        println!(
            "\n  Best model: {} (score {:.3}, {}/{} passed)",
            winner.model.bold().green(),
            winner.avg_score,
            winner.passed,
            winner.total,
        );
    }

    // Warn when the gap between best and worst is significant
    if outcomes.len() >= 2 {
        let best = outcomes[0].avg_score;
        let worst = outcomes.last().unwrap().avg_score;
        let gap = best - worst;
        if gap > 0.1 {
            println!(
                "  {} Score spread of {:.1}% — models differ significantly on this suite.",
                "⚠".yellow(),
                gap * 100.0,
            );
        }
    }

    println!();
    Ok(())
}
