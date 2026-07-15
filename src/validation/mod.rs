pub mod judge_validator;

use anyhow::Result;
use colored::Colorize;
use crate::cli::ValidateJudgeArgs;
use crate::embedded;
pub use judge_validator::validate_judge;

pub async fn run(args: ValidateJudgeArgs) -> Result<()> {
    // Resolve suite paths
    let suite_paths = if args.suite == "safety" {
        // Load all safety suites
        vec![
            "owasp_llm01_injection.toml",
            "owasp_llm02_sensitive_disclosure.toml",
            "owasp_llm03_supply_chain.toml",
            "owasp_llm04_data_poisoning.toml",
            "owasp_llm05_output_handling.toml",
            "owasp_llm06_excessive_agency.toml",
            "owasp_llm07_system_prompt_leakage.toml",
            "owasp_llm08_vector_weaknesses.toml",
            "owasp_llm09_misinformation.toml",
            "owasp_llm10_unbounded_consumption.toml",
        ]
        .into_iter()
        .map(|name| embedded::resolve_suite_path(&format!("safety/{name}")))
        .collect()
    } else {
        vec![embedded::resolve_suite_path(&args.suite)]
    };

    // Parse judges
    let judges = args.judges.unwrap_or_else(|| {
        "anthropic:claude-fable-5,anthropic:claude-3-5-haiku-latest,groq:llama-3.1-8b-instant".to_string()
    });
    let judge_list: Vec<String> = judges.split(',').map(|s| s.trim().to_string()).collect();

    // Run validation
    let report = validate_judge(suite_paths, judge_list).await?;

    // Display results
    println!("\n{} Results", "═".repeat(68).green().bold());
    println!("  Total tests analyzed: {}", report.total_tests);
    println!("  Judges compared:      {}", report.judges_compared.len());
    println!("  Overall agreement:    {:.1}%", report.overall_agreement);
    println!("  Confidence level:     {}", report.summary.confidence_level.yellow());
    println!("  Reliable:             {}", if report.summary.is_reliable { "✓ YES".green() } else { "✗ NO".red() });

    if !report.per_rubric_metrics.is_empty() {
        println!("\n{} Per-Rubric Metrics", "─".repeat(68).cyan());
        for metric in report.per_rubric_metrics.values() {
            println!(
                "  {}: {:.1}% agreement ({} tests, {} disagreements)",
                metric.rubric_name.cyan(),
                metric.agreement_percentage,
                metric.test_count,
                metric.disagreement_count
            );
        }
    }

    if args.show_divergent && !report.divergent_cases.is_empty() {
        println!("\n{} Divergent Cases", "⚠".repeat(35).yellow());
        for case in &report.divergent_cases {
            println!("  Test: {} ({})", case.test_name.yellow(), case.rubric.dimmed());
            for (judge, score) in &case.judge_scores {
                println!("    {}: {:.2}", judge, score);
            }
            println!("    Divergence: {:.2}", format!("{:.2}", case.max_disagreement).red());
        }
    }

    if !report.summary.recommendations.is_empty() {
        println!("\n{} Recommendations", "💡".cyan());
        for rec in &report.summary.recommendations {
            println!("  • {}", rec);
        }
    }

    println!();
    Ok(())
}
