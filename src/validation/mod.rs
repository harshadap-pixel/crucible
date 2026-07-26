pub mod baseline;
pub mod judge_validator;

use anyhow::Result;
use colored::Colorize;

use crate::cli::ValidateJudgeArgs;
use crate::embedded;

pub use baseline::BaselineManager;
pub use judge_validator::{
    compute_metrics, generate_summary, score_output_with_judge, validate_judge, FallbackConfig,
};

pub async fn run(args: ValidateJudgeArgs) -> Result<()> {
    // Resolve suite path
    let suite_path = if args.suite == "safety" {
        // Default to first safety suite for demo
        embedded::resolve_suite_path("safety/owasp_llm01_injection.toml")
    } else {
        embedded::resolve_suite_path(&args.suite)
    };

    // Build fallback config first (before moving args fields)
    let fallback_config = FallbackConfig::from_cli_args(&args);

    // Parse judges
    let judges = args.judges.unwrap_or_else(|| {
        "anthropic:claude-fable-5,anthropic:claude-3-5-haiku-latest,groq:llama-3.1-8b-instant"
            .to_string()
    });
    let judge_list: Vec<String> = judges.split(',').map(|s| s.trim().to_string()).collect();

    // Use specified model or default to llama3.1:8b
    let eval_model = args.model.as_deref().unwrap_or("llama3.1:8b");

    // Run validation
    let report = validate_judge(&suite_path, eval_model, judge_list, fallback_config).await?;

    // Display results
    println!("\n{} Validation Results", "═".repeat(65).green().bold());
    println!("  Suite:              {}", report.suite_name.cyan());
    println!("  Tests analyzed:     {}", report.tests_with_judges);
    println!("  Judges compared:    {}", report.judges_compared.len());
    println!("  Overall agreement:  {:.1}%", report.overall_agreement);
    println!(
        "  Confidence:         {}",
        match report.summary.confidence_level.as_str() {
            "high" => report.summary.confidence_level.green(),
            "medium" => report.summary.confidence_level.yellow(),
            _ => report.summary.confidence_level.red(),
        }
    );
    println!(
        "  Reliable:           {}",
        if report.summary.is_reliable {
            "✓ YES".green()
        } else {
            "✗ NO".red()
        }
    );

    // Per-rubric metrics
    if !report.per_rubric_metrics.is_empty() {
        println!("\n{} Per-Rubric Metrics", "─".repeat(65).cyan());
        for metric in report.per_rubric_metrics.values() {
            println!(
                "  {}: {:.1}% ({} cases, {} divergent)",
                metric.rubric_name.cyan(),
                metric.agreement_percentage,
                metric.test_count,
                metric.disagreement_count
            );
        }
    }

    // Divergent cases
    if args.show_divergent && !report.divergent_cases.is_empty() {
        println!("\n{} Divergent Cases (agreement gap > 15%)", "⚠".yellow());
        for case in &report.divergent_cases {
            println!(
                "  {}: {} → divergence {}",
                case.test_name.yellow(),
                case.rubric.dimmed(),
                format!("{:.2}", case.max_disagreement).red()
            );
        }
    }

    // Recommendations
    if !report.summary.recommendations.is_empty() {
        println!("\n{} Recommendations", "💡".cyan());
        for rec in &report.summary.recommendations {
            println!("  • {}", rec);
        }
    }

    println!();
    Ok(())
}
