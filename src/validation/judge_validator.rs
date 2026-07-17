use anyhow::Result;
use colored::Colorize;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use crate::providers::{ModelRef, OllamaClient};
use crate::runner;
use crate::cli::RunArgs;
use indicatif::{ProgressBar, ProgressStyle};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct JudgeValidationReport {
    pub suite_name: String,
    pub total_tests: usize,
    pub tests_with_judges: usize,
    pub judges_compared: Vec<String>,
    pub overall_agreement: f64,
    pub per_rubric_metrics: HashMap<String, RubricMetrics>,
    pub divergent_cases: Vec<DivergentCase>,
    pub summary: ValidationSummary,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RubricMetrics {
    pub rubric_name: String,
    pub test_count: usize,
    pub agreement_percentage: f64,
    pub disagreement_count: usize,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DivergentCase {
    pub test_name: String,
    pub rubric: String,
    pub judge_scores: HashMap<String, f64>,
    pub max_disagreement: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ValidationSummary {
    pub is_reliable: bool,
    pub confidence_level: String,
    pub recommendations: Vec<String>,
}

// TODO: Implementation stubs
// These will be implemented in Phase 2-4

/// Score a model output against a rubric using a specific judge model.
/// Returns the raw score (0.0-1.0) without comparing to threshold.
pub async fn score_output_with_judge(
    judge: &ModelRef,
    output: &str,
    rubric: &str,
) -> Result<f64> {
    let prompt = format!(
        "RUBRIC:\n{}\n\nRESPONSE TO EVALUATE:\n{}\n\nScore the response.",
        rubric, output
    );

    let system = "You are a strict evaluator. Score from 0.0 to 1.0. Reply with ONLY a JSON object: {\"score\": <float>, \"reason\": \"<one sentence>\"}. No other text.";

    match judge.provider.chat(&judge.model, Some(system), &prompt, 0.0).await {
        Ok(result) => {
            if let Some((score, _)) = parse_judge_response(&result.text) {
                Ok(score)
            } else {
                eprintln!(
                    "  {} Failed to parse judge response from {}",
                    "⚠".yellow(),
                    judge.model
                );
                Ok(0.5) // Fallback score
            }
        }
        Err(e) => {
            eprintln!(
                "  {} Judge {} failed: {}",
                "⚠".yellow(),
                judge.model,
                e
            );
            Ok(0.0) // Fallback score for failures
        }
    }
}

/// Parse judge response JSON to extract score and reason.
fn parse_judge_response(text: &str) -> Option<(f64, String)> {
    let start = text.find('{')?;
    let end = text.rfind('}')? + 1;
    let json_str = &text[start..end];

    let v: serde_json::Value = serde_json::from_str(json_str).ok()?;
    let score = v.get("score")?.as_f64()?;
    let reason = v.get("reason")?.as_str()?.to_string();
    Some((score.clamp(0.0, 1.0), reason))
}

/// Compute agreement metrics from judge scores.
pub fn compute_metrics(
    judgements: &[(String, String, Vec<f64>, Vec<String>)], // (test_name, rubric, scores, judge_names)
) -> (f64, HashMap<String, RubricMetrics>, Vec<DivergentCase>) {
    let mut per_rubric_data: HashMap<String, Vec<Vec<f64>>> = HashMap::new();
    let mut divergent_cases = Vec::new();

    // Group scores by rubric
    for (test_name, rubric, scores, judge_names) in judgements {
        per_rubric_data
            .entry(rubric.clone())
            .or_default()
            .push(scores.clone());

        // Detect divergence
        if scores.len() > 1 {
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let min = scores.iter().copied().fold(f64::INFINITY, f64::min);
            let disagreement = max - min;

            if disagreement > 0.15 {
                // Populate judge_scores with actual judge names
                let mut judge_scores = HashMap::new();
                for (judge_name, score) in judge_names.iter().zip(scores.iter()) {
                    judge_scores.insert(judge_name.clone(), *score);
                }

                divergent_cases.push(DivergentCase {
                    test_name: test_name.clone(),
                    rubric: rubric.clone(),
                    judge_scores,
                    max_disagreement: disagreement,
                });
            }
        }
    }

    // Compute per-rubric metrics
    let mut per_rubric_metrics = HashMap::new();
    let mut total_agreement = 0.0;
    let mut total_rubrics = 0;

    for (rubric, all_score_sets) in per_rubric_data {
        let mut rubric_agreement = 0.0;
        let mut disagreement_count = 0;

        for scores in &all_score_sets {
            if scores.len() > 1 {
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let min = scores.iter().copied().fold(f64::INFINITY, f64::min);
                let disagreement = max - min;
                let agreement = 1.0 - disagreement.min(1.0);
                rubric_agreement += agreement;

                if disagreement > 0.15 {
                    disagreement_count += 1;
                }
            } else {
                rubric_agreement += 1.0;
            }
        }

        let agreement_pct = if all_score_sets.is_empty() {
            100.0
        } else {
            (rubric_agreement / all_score_sets.len() as f64) * 100.0
        };

        per_rubric_metrics.insert(
            rubric.clone(),
            RubricMetrics {
                rubric_name: rubric,
                test_count: all_score_sets.len(),
                agreement_percentage: agreement_pct,
                disagreement_count,
            },
        );

        total_agreement += agreement_pct;
        total_rubrics += 1;
    }

    let overall_agreement = if total_rubrics == 0 {
        100.0
    } else {
        total_agreement / total_rubrics as f64
    };

    (overall_agreement, per_rubric_metrics, divergent_cases)
}

/// Generate validation report summary.
pub fn generate_summary(overall_agreement: f64, divergent_count: usize, total_tests: usize) -> ValidationSummary {
    let confidence_level = match overall_agreement {
        a if a >= 90.0 => "high".to_string(),
        a if a >= 75.0 => "medium".to_string(),
        _ => "low".to_string(),
    };

    let is_reliable = overall_agreement >= 85.0;

    let mut recommendations = vec![];
    if !is_reliable {
        recommendations.push("Judge agreement is below 85% — investigate divergent cases".to_string());
    }
    if divergent_count > (total_tests / 10).max(1) {
        recommendations.push(format!(
            "High divergence rate ({} cases) — consider rubric clarification",
            divergent_count
        ));
    }
    if recommendations.is_empty() {
        recommendations.push("Judge performance is acceptable".to_string());
    }

    ValidationSummary {
        is_reliable,
        confidence_level,
        recommendations,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_metrics_perfect_agreement() {
        // All judges agree perfectly
        let judgements = vec![
            (
                "test1".to_string(),
                "rubric1".to_string(),
                vec![0.85, 0.85, 0.85],
                vec!["judge_a".to_string(), "judge_b".to_string(), "judge_c".to_string()],
            ),
        ];

        let (overall, _metrics, divergent) = compute_metrics(&judgements);

        assert_eq!(overall, 100.0); // Perfect agreement
        assert_eq!(divergent.len(), 0); // No divergent cases
    }

    #[test]
    fn test_compute_metrics_with_divergence() {
        // Judges disagree significantly
        let judgements = vec![
            (
                "test1".to_string(),
                "rubric1".to_string(),
                vec![0.9, 0.5, 0.3], // Large gaps
                vec!["judge_a".to_string(), "judge_b".to_string(), "judge_c".to_string()],
            ),
        ];

        let (overall, _metrics, divergent) = compute_metrics(&judgements);

        assert!(overall < 100.0); // Not perfect
        assert_eq!(divergent.len(), 1); // Detected divergence
        assert_eq!(divergent[0].test_name, "test1");
        assert!(divergent[0].max_disagreement > 0.15);
    }

    #[test]
    fn test_compute_metrics_judge_scores_populated() {
        // Verify judge names are properly mapped to scores
        let judgements = vec![
            (
                "test1".to_string(),
                "rubric1".to_string(),
                vec![0.9, 0.4],
                vec!["fable".to_string(), "haiku".to_string()],
            ),
        ];

        let (_overall, _metrics, divergent) = compute_metrics(&judgements);

        assert_eq!(divergent.len(), 1);
        assert_eq!(divergent[0].judge_scores.get("fable"), Some(&0.9));
        assert_eq!(divergent[0].judge_scores.get("haiku"), Some(&0.4));
    }

    #[test]
    fn test_generate_summary_high_confidence() {
        let summary = generate_summary(92.0, 0, 10);

        assert!(summary.is_reliable);
        assert_eq!(summary.confidence_level, "high");
    }

    #[test]
    fn test_generate_summary_medium_confidence() {
        let summary = generate_summary(80.0, 0, 10);

        // 80% agreement is below 85% threshold, so NOT reliable
        assert!(!summary.is_reliable);
        assert_eq!(summary.confidence_level, "medium");
    }

    #[test]
    fn test_generate_summary_low_confidence() {
        let summary = generate_summary(70.0, 5, 10);

        assert!(!summary.is_reliable);
        assert_eq!(summary.confidence_level, "low");
    }

    #[test]
    fn test_generate_summary_high_divergence_rate() {
        // 90% agreement (reliable) but very high divergence rate (5/10 = 50%)
        // should trigger divergence recommendation
        let summary = generate_summary(90.0, 5, 10);

        // 90% >= 85%, so technically reliable, but high divergence rate
        assert!(summary.is_reliable);
        assert!(summary
            .recommendations
            .iter()
            .any(|r| r.contains("High divergence")));
    }

    #[test]
    fn test_parse_judge_response_valid() {
        let json = r#"{"score": 0.85, "reason": "Good response"}"#;
        let (score, reason) = parse_judge_response(json).expect("Should parse");

        assert_eq!(score, 0.85);
        assert_eq!(reason, "Good response");
    }

    #[test]
    fn test_parse_judge_response_with_extra_text() {
        let json = r#"Some text {"score": 0.75, "reason": "OK"} more text"#;
        let (score, reason) = parse_judge_response(json).expect("Should parse");

        assert_eq!(score, 0.75);
        assert_eq!(reason, "OK");
    }

    #[test]
    fn test_parse_judge_response_clamped() {
        // Score > 1.0 should be clamped
        let json = r#"{"score": 1.5, "reason": "Invalid"}"#;
        let (score, _) = parse_judge_response(json).expect("Should parse");

        assert_eq!(score, 1.0);
    }

    #[test]
    fn test_parse_judge_response_invalid() {
        let json = r#"{"invalid": "format"}"#;
        assert!(parse_judge_response(json).is_none());
    }
}

/// Main validation orchestrator — Phase 3 Complete
/// 1. Run test suite with specified model to get actual outputs
/// 2. Extract tests with llm_judge assertions
/// 3. For each test output, collect scores from multiple judges
/// 4. Compute agreement metrics
/// 5. Return comprehensive report
pub async fn validate_judge(
    suite_path: &str,
    model: &str,
    judges: Vec<String>,
) -> Result<JudgeValidationReport> {
    println!("\n{}", "Judge Validation".cyan().bold());
    println!("{}", "─".repeat(70).dimmed());
    println!("  Suite:  {}", suite_path.yellow());
    println!("  Model:  {}", model.yellow());
    println!("  Judges: {}", judges.join(", ").yellow());
    println!();

    // Parse judge model specs
    let judge_refs: Vec<ModelRef> = judges
        .iter()
        .map(|spec| ModelRef::resolve(spec, "http://localhost:11434"))
        .collect::<Result<Vec<_>>>()?;

    // ── Phase 3: Run test suite to get actual outputs ──
    println!("  {} Running test suite...", "▸".cyan());

    // Create RunArgs for test execution
    let run_args = RunArgs {
        suite: suite_path.to_string(),
        model: Some(model.to_string()),
        n_runs: 1,
        concurrency: 2,
        ollama_url: "http://localhost:11434".to_string(),
        dir: None,
        dataset: None,
        template: None,
        slice_by: None,
        category: None,
        filter: None,
        vars: Vec::new(),
        judge: None,
        models: Vec::new(),
        fail_fast: false,
        retry: 0,
        output: "terminal".to_string(),
        compare: false,
        baseline: false,
        update_snapshots: false,
    };

    // Create Ollama client for embeddings
    let embed_client = Arc::new(OllamaClient::new(&run_args.ollama_url));

    // Execute the suite to get test results with outputs
    let outcome = runner::execute_suite(
        suite_path,
        None,
        &run_args,
        &HashMap::new(),
        &embed_client,
    )
    .await?;

    println!("  {} Executed {} tests", "✓".green(), outcome.total);

    // ── Extract tests with llm_judge assertions ──
    let suite_content = std::fs::read_to_string(suite_path)?;
    let suite: crate::config::Suite = toml::from_str(&suite_content)?;

    let mut test_judge_pairs: Vec<(String, String, String)> = Vec::new(); // (test_name, output, rubric)

    for test in &suite.tests {
        for assertion in &test.assert {
            if let crate::config::Assertion::LlmJudge { rubric, .. } = assertion {
                // Find corresponding test result
                if let Some(result) = outcome.results.iter().find(|r| r.test_name == test.name) {
                    test_judge_pairs.push((
                        test.name.clone(),
                        result.output.clone(),
                        rubric.clone(),
                    ));
                }
            }
        }
    }

    if test_judge_pairs.is_empty() {
        println!(
            "  {} No llm_judge assertions found in suite",
            "ℹ".cyan()
        );
        return Ok(JudgeValidationReport {
            suite_name: suite.suite.name.clone(),
            total_tests: outcome.total as usize,
            tests_with_judges: 0,
            judges_compared: judges,
            overall_agreement: 100.0,
            per_rubric_metrics: HashMap::new(),
            divergent_cases: vec![],
            summary: ValidationSummary {
                is_reliable: true,
                confidence_level: "high".to_string(),
                recommendations: vec!["No judge assertions to validate".to_string()],
            },
        });
    }

    println!("  {} Found {} judge assertions", "✓".green(), test_judge_pairs.len());
    println!();

    // ── Phase 3: Collect judge scores for each output ──
    println!("  {} Collecting judge scores...", "▸".cyan());
    let pb = ProgressBar::new(test_judge_pairs.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.cyan} [{bar:20.cyan/blue}] {pos}/{len}")
            .unwrap()
    );

    let mut judgements: Vec<(String, String, Vec<f64>, Vec<String>)> = Vec::new();
    let judge_names: Vec<String> = judge_refs.iter().map(|j| j.model.clone()).collect();

    for (test_name, output, rubric) in test_judge_pairs {
        pb.inc(1);

        let mut scores = Vec::new();

        // Run output through each judge
        for judge in &judge_refs {
            match score_output_with_judge(judge, &output, &rubric).await {
                Ok(score) => scores.push(score),
                Err(e) => {
                    eprintln!("  {} Judge {} failed: {}", "⚠".yellow(), judge.model, e);
                    scores.push(0.5); // Fallback score
                }
            }
        }

        judgements.push((test_name, rubric, scores, judge_names.clone()));
    }
    pb.finish_with_message("✓ Collected");
    println!();

    // Compute metrics
    let (overall_agreement, per_rubric_metrics, divergent_cases) = compute_metrics(&judgements);
    let summary = generate_summary(overall_agreement, divergent_cases.len(), judgements.len());

    Ok(JudgeValidationReport {
        suite_name: suite.suite.name.clone(),
        total_tests: outcome.total as usize,
        tests_with_judges: judgements.len(),
        judges_compared: judges,
        overall_agreement,
        per_rubric_metrics,
        divergent_cases,
        summary,
    })
}
