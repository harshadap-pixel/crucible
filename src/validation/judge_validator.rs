use anyhow::Result;
use colored::Colorize;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use crate::cli::RunArgs;
use crate::providers::{ModelRef, OllamaClient};
use crate::runner;
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub regression_data: Option<RegressionData>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RegressionData {
    pub baseline_agreement: f64,
    pub agreement_delta: f64, // Current - baseline (negative = regression)
    pub is_regression: bool,  // True if delta < -5%
    pub judge_degradation: HashMap<String, JudgeDegradation>,
    pub trend_direction: TrendDirection,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct JudgeDegradation {
    pub judge_name: String,
    pub previous_agreement: f64,
    pub current_agreement: f64,
    pub degradation_rate: f64, // Previous - current (positive = degradation)
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub enum TrendDirection {
    Improving, // Agreement increasing over time
    Stable,    // Agreement within ±2%
    Degrading, // Agreement decreasing over time
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct JudgeValidationBaseline {
    pub timestamp: String,
    pub suite_name: String,
    pub overall_agreement: f64,
    pub per_judge_agreement: HashMap<String, f64>,
    pub per_rubric_metrics: HashMap<String, RubricMetrics>,
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

#[derive(Debug, Clone)]
pub struct FallbackConfig {
    pub strategy: FallbackStrategy,
    pub auth_score: f64,
    pub timeout_score: f64,
    pub error_score: f64,
    pub fallback_judges: Vec<String>,
    pub skip_on_all_fail: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FallbackStrategy {
    /// Use severity-based scores (0.0 for auth, 0.5 for timeout/generic)
    Severity,
    /// Skip test if all judges fail
    Skip,
    /// Use average of working judges
    Average,
    /// Try primary judges, then fallback judges in order
    PrimaryWithFallback,
}

/// Compute regression metrics by comparing current results against baseline
pub fn compute_regression(
    current: &JudgeValidationReport,
    baseline: &JudgeValidationBaseline,
) -> RegressionData {
    let agreement_delta = current.overall_agreement - baseline.overall_agreement;
    let is_regression = agreement_delta < -5.0; // 5% drop = regression

    // Compute judge-by-judge degradation
    let mut judge_degradation = HashMap::new();
    for judge in &current.judges_compared {
        if let Some(baseline_agreement) = baseline.per_judge_agreement.get(judge) {
            let current_agreement = current.overall_agreement; // Simplified: use overall
            let degradation_rate = baseline_agreement - current_agreement;

            judge_degradation.insert(
                judge.clone(),
                JudgeDegradation {
                    judge_name: judge.clone(),
                    previous_agreement: *baseline_agreement,
                    current_agreement,
                    degradation_rate,
                },
            );
        }
    }

    // Determine trend direction
    let trend_direction = if agreement_delta > 2.0 {
        TrendDirection::Improving
    } else if agreement_delta < -2.0 {
        TrendDirection::Degrading
    } else {
        TrendDirection::Stable
    };

    RegressionData {
        baseline_agreement: baseline.overall_agreement,
        agreement_delta,
        is_regression,
        judge_degradation,
        trend_direction,
    }
}

impl FallbackConfig {
    pub fn from_cli_args(args: &crate::cli::ValidateJudgeArgs) -> Self {
        let strategy = match args.fallback_strategy.as_str() {
            "skip" => FallbackStrategy::Skip,
            "average" => FallbackStrategy::Average,
            "primary+fallback" => FallbackStrategy::PrimaryWithFallback,
            _ => FallbackStrategy::Severity,
        };

        let fallback_judges = args
            .fallback_judges
            .as_ref()
            .map(|j| j.split(',').map(|s| s.trim().to_string()).collect())
            .unwrap_or_default();

        Self {
            strategy,
            auth_score: args.fallback_auth_score,
            timeout_score: args.fallback_timeout_score,
            error_score: args.fallback_error_score,
            fallback_judges,
            skip_on_all_fail: args.skip_on_all_fail,
        }
    }

    /// Determine fallback score based on error type and strategy
    pub fn get_fallback_score(&self, error_msg: &str) -> Option<f64> {
        match self.strategy {
            FallbackStrategy::Severity => {
                if error_msg.contains("401") || error_msg.contains("Unauthorized") {
                    Some(self.auth_score)
                } else if error_msg.contains("timeout") || error_msg.contains("timed out") {
                    Some(self.timeout_score)
                } else {
                    Some(self.error_score)
                }
            }
            FallbackStrategy::Skip => None, // Signal to skip this test
            FallbackStrategy::Average => None, // Will be computed from working judges
            FallbackStrategy::PrimaryWithFallback => None, // Will try fallback judges
        }
    }
}

/// Score a model output against a rubric using a specific judge model.
/// Returns the raw score (0.0-1.0) without comparing to threshold.
/// Truncates long outputs to avoid token limits; propagates errors for caller's error handling.
pub async fn score_output_with_judge(judge: &ModelRef, output: &str, rubric: &str) -> Result<f64> {
    // Truncate very long outputs to avoid token limits
    let max_output_len = 2000;
    let truncated_output = if output.len() > max_output_len {
        format!("{}...[truncated]", &output[..max_output_len])
    } else {
        output.to_string()
    };

    let prompt = format!(
        "RUBRIC:\n{}\n\nRESPONSE TO EVALUATE:\n{}\n\nScore the response.",
        rubric, truncated_output
    );

    let system = "You are a strict evaluator. Score from 0.0 to 1.0. Reply with ONLY a JSON object: {\"score\": <float>, \"reason\": \"<one sentence>\"}. No other text.";

    // Call the judge model — propagate errors so caller can handle them
    let result = judge
        .provider
        .chat(&judge.model, Some(system), &prompt, 0.0)
        .await?;

    // Parse the response
    match parse_judge_response(&result.text) {
        Some((score, _reason)) => Ok(score),
        None => {
            // Response exists but is malformed
            anyhow::bail!(
                "Judge {} returned unparseable JSON: {}",
                judge.model,
                &result.text[..result.text.len().min(100)]
            );
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
pub fn generate_summary(
    overall_agreement: f64,
    divergent_count: usize,
    total_tests: usize,
) -> ValidationSummary {
    let confidence_level = match overall_agreement {
        a if a >= 90.0 => "high".to_string(),
        a if a >= 75.0 => "medium".to_string(),
        _ => "low".to_string(),
    };

    let is_reliable = overall_agreement >= 85.0;

    let mut recommendations = vec![];
    if !is_reliable {
        recommendations
            .push("Judge agreement is below 85% — investigate divergent cases".to_string());
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
#[allow(clippy::items_after_test_module, unused_variables)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_metrics_perfect_agreement() {
        // All judges agree perfectly
        let judgements = vec![(
            "test1".to_string(),
            "rubric1".to_string(),
            vec![0.85, 0.85, 0.85],
            vec![
                "judge_a".to_string(),
                "judge_b".to_string(),
                "judge_c".to_string(),
            ],
        )];

        let (overall, _metrics, divergent) = compute_metrics(&judgements);

        assert_eq!(overall, 100.0); // Perfect agreement
        assert_eq!(divergent.len(), 0); // No divergent cases
    }

    #[test]
    fn test_compute_metrics_with_divergence() {
        // Judges disagree significantly
        let judgements = vec![(
            "test1".to_string(),
            "rubric1".to_string(),
            vec![0.9, 0.5, 0.3], // Large gaps
            vec![
                "judge_a".to_string(),
                "judge_b".to_string(),
                "judge_c".to_string(),
            ],
        )];

        let (overall, _metrics, divergent) = compute_metrics(&judgements);

        assert!(overall < 100.0); // Not perfect
        assert_eq!(divergent.len(), 1); // Detected divergence
        assert_eq!(divergent[0].test_name, "test1");
        assert!(divergent[0].max_disagreement > 0.15);
    }

    #[test]
    fn test_compute_metrics_judge_scores_populated() {
        // Verify judge names are properly mapped to scores
        let judgements = vec![(
            "test1".to_string(),
            "rubric1".to_string(),
            vec![0.9, 0.4],
            vec!["fable".to_string(), "haiku".to_string()],
        )];

        let (overall, metrics, divergent) = compute_metrics(&judgements);

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

    // ── Fallback Strategy Tests ──────────────────────────────────────────────

    #[test]
    fn test_fallback_severity_auth_error() {
        let config = FallbackConfig {
            strategy: FallbackStrategy::Severity,
            auth_score: 0.0,
            timeout_score: 0.5,
            error_score: 0.5,
            fallback_judges: vec![],
            skip_on_all_fail: false,
        };

        let score = config.get_fallback_score("401 Unauthorized");
        assert_eq!(score, Some(0.0));
    }

    #[test]
    fn test_fallback_severity_timeout_error() {
        let config = FallbackConfig {
            strategy: FallbackStrategy::Severity,
            auth_score: 0.0,
            timeout_score: 0.5,
            error_score: 0.4,
            fallback_judges: vec![],
            skip_on_all_fail: false,
        };

        let score = config.get_fallback_score("request timed out");
        assert_eq!(score, Some(0.5));
    }

    #[test]
    fn test_fallback_severity_generic_error() {
        let config = FallbackConfig {
            strategy: FallbackStrategy::Severity,
            auth_score: 0.0,
            timeout_score: 0.5,
            error_score: 0.3,
            fallback_judges: vec![],
            skip_on_all_fail: false,
        };

        let score = config.get_fallback_score("some other error");
        assert_eq!(score, Some(0.3));
    }

    #[test]
    fn test_fallback_skip_strategy_returns_none() {
        let config = FallbackConfig {
            strategy: FallbackStrategy::Skip,
            auth_score: 0.0,
            timeout_score: 0.5,
            error_score: 0.5,
            fallback_judges: vec![],
            skip_on_all_fail: false,
        };

        // Skip strategy should return None for any error
        assert_eq!(config.get_fallback_score("401 Unauthorized"), None);
        assert_eq!(config.get_fallback_score("timeout"), None);
        assert_eq!(config.get_fallback_score("any error"), None);
    }

    #[test]
    fn test_fallback_average_strategy_returns_none() {
        let config = FallbackConfig {
            strategy: FallbackStrategy::Average,
            auth_score: 0.0,
            timeout_score: 0.5,
            error_score: 0.5,
            fallback_judges: vec![],
            skip_on_all_fail: false,
        };

        // Average strategy should return None (computed later from working judges)
        assert_eq!(config.get_fallback_score("any error"), None);
    }

    #[test]
    fn test_fallback_primary_with_fallback_returns_none() {
        let config = FallbackConfig {
            strategy: FallbackStrategy::PrimaryWithFallback,
            auth_score: 0.0,
            timeout_score: 0.5,
            error_score: 0.5,
            fallback_judges: vec!["haiku".to_string()],
            skip_on_all_fail: false,
        };

        // PrimaryWithFallback should return None (tries fallback judge instead)
        assert_eq!(config.get_fallback_score("any error"), None);
    }

    #[test]
    fn test_fallback_custom_scores() {
        let config = FallbackConfig {
            strategy: FallbackStrategy::Severity,
            auth_score: 0.2,    // Custom auth score
            timeout_score: 0.7, // Custom timeout score
            error_score: 0.6,   // Custom error score
            fallback_judges: vec![],
            skip_on_all_fail: false,
        };

        assert_eq!(config.get_fallback_score("401"), Some(0.2));
        assert_eq!(config.get_fallback_score("timeout"), Some(0.7));
        assert_eq!(config.get_fallback_score("other"), Some(0.6));
    }

    #[test]
    fn test_fallback_config_from_defaults() {
        // Verify default fallback config values
        let cli_args = crate::cli::ValidateJudgeArgs {
            suite: "default.toml".to_string(),
            model: None,
            judges: None,
            show_divergent: false,
            fallback_strategy: "severity".to_string(),
            fallback_auth_score: 0.0,
            fallback_timeout_score: 0.5,
            fallback_error_score: 0.5,
            fallback_judges: None,
            skip_on_all_fail: false,
            ollama_url: "http://localhost:11434".to_string(),
        };

        let config = FallbackConfig::from_cli_args(&cli_args);
        assert_eq!(config.strategy, FallbackStrategy::Severity);
        assert_eq!(config.auth_score, 0.0);
        assert_eq!(config.timeout_score, 0.5);
        assert_eq!(config.error_score, 0.5);
        assert!(!config.skip_on_all_fail);
    }

    #[test]
    fn test_fallback_config_skip_on_all_fail() {
        let cli_args = crate::cli::ValidateJudgeArgs {
            suite: "default.toml".to_string(),
            model: None,
            judges: None,
            show_divergent: false,
            fallback_strategy: "skip".to_string(),
            fallback_auth_score: 0.0,
            fallback_timeout_score: 0.5,
            fallback_error_score: 0.5,
            fallback_judges: None,
            skip_on_all_fail: true,
            ollama_url: "http://localhost:11434".to_string(),
        };

        let config = FallbackConfig::from_cli_args(&cli_args);
        assert_eq!(config.strategy, FallbackStrategy::Skip);
        assert!(config.skip_on_all_fail);
    }

    #[test]
    fn test_fallback_judges_parsing() {
        let cli_args = crate::cli::ValidateJudgeArgs {
            suite: "default.toml".to_string(),
            model: None,
            judges: None,
            show_divergent: false,
            fallback_strategy: "primary+fallback".to_string(),
            fallback_auth_score: 0.0,
            fallback_timeout_score: 0.5,
            fallback_error_score: 0.5,
            fallback_judges: Some("haiku,llama-3.1-8b".to_string()),
            skip_on_all_fail: false,
            ollama_url: "http://localhost:11434".to_string(),
        };

        let config = FallbackConfig::from_cli_args(&cli_args);
        assert_eq!(config.strategy, FallbackStrategy::PrimaryWithFallback);
        assert_eq!(config.fallback_judges, vec!["haiku", "llama-3.1-8b"]);
    }

    // ── Metrics Edge Cases ───────────────────────────────────────────────────

    #[test]
    fn test_metrics_empty_judgements() {
        // No tests at all
        let judgements: Vec<(String, String, Vec<f64>, Vec<String>)> = vec![];

        let (overall, metrics, divergent) = compute_metrics(&judgements);

        assert_eq!(overall, 100.0); // Empty = perfect
        assert_eq!(metrics.len(), 0);
        assert_eq!(divergent.len(), 0);
    }

    #[test]
    fn test_metrics_single_test_single_judge() {
        // Only one test with one judge (can't compute agreement)
        let judgements = vec![(
            "test1".to_string(),
            "rubric1".to_string(),
            vec![0.85],
            vec!["judge_a".to_string()],
        )];

        let (overall, metrics, divergent) = compute_metrics(&judgements);

        assert_eq!(overall, 100.0); // Single judge treated as perfect agreement
        assert_eq!(metrics["rubric1"].agreement_percentage, 100.0);
        assert_eq!(metrics["rubric1"].test_count, 1);
        assert_eq!(divergent.len(), 0); // Can't diverge with one judge
    }

    #[test]
    fn test_metrics_multiple_tests_single_judge() {
        // Multiple tests, but only one judge (can't compute agreement)
        let judgements = vec![
            (
                "test1".to_string(),
                "rubric1".to_string(),
                vec![0.85],
                vec!["judge_a".to_string()],
            ),
            (
                "test2".to_string(),
                "rubric1".to_string(),
                vec![0.75],
                vec!["judge_a".to_string()],
            ),
        ];

        let (overall, metrics, divergent) = compute_metrics(&judgements);

        assert_eq!(overall, 100.0); // Single judge = perfect
        assert_eq!(metrics["rubric1"].test_count, 2);
        assert_eq!(divergent.len(), 0);
    }

    #[test]
    fn test_metrics_empty_score_vectors() {
        // Edge case: empty score vector (should not happen but test robustness)
        let judgements = vec![(
            "test1".to_string(),
            "rubric1".to_string(),
            vec![], // Empty!
            vec![],
        )];

        let (overall, metrics, divergent) = compute_metrics(&judgements);

        // Empty scores treated as single judge (perfect agreement)
        assert_eq!(overall, 100.0);
        assert_eq!(metrics["rubric1"].agreement_percentage, 100.0);
        assert_eq!(divergent.len(), 0);
    }

    #[test]
    fn test_metrics_perfect_agreement_all_judges() {
        // All judges agree perfectly (0.85, 0.85, 0.85)
        let judgements = vec![(
            "test1".to_string(),
            "rubric1".to_string(),
            vec![0.85, 0.85, 0.85],
            vec![
                "judge_a".to_string(),
                "judge_b".to_string(),
                "judge_c".to_string(),
            ],
        )];

        let (overall, metrics, divergent) = compute_metrics(&judgements);

        assert_eq!(overall, 100.0);
        assert_eq!(metrics["rubric1"].agreement_percentage, 100.0);
        assert_eq!(metrics["rubric1"].disagreement_count, 0);
        assert_eq!(divergent.len(), 0); // No divergence
    }

    #[test]
    fn test_metrics_boundary_divergence_just_under_15_percent() {
        // Scores differ by 0.149 (boundary: should NOT flag as divergent)
        let judgements = vec![(
            "test1".to_string(),
            "rubric1".to_string(),
            vec![0.85, 0.701], // Difference = 0.149
            vec!["judge_a".to_string(), "judge_b".to_string()],
        )];

        let (overall, metrics, divergent) = compute_metrics(&judgements);

        assert!(overall < 100.0); // Not perfect agreement
        assert_eq!(divergent.len(), 0); // 0.149 is NOT > 0.15, so no divergent case
    }

    #[test]
    fn test_metrics_boundary_divergence_slightly_above_15_percent() {
        // Scores differ by 0.151 (boundary: SHOULD flag as divergent)
        let judgements = vec![(
            "test1".to_string(),
            "rubric1".to_string(),
            vec![0.85, 0.699], // Difference = 0.151
            vec!["judge_a".to_string(), "judge_b".to_string()],
        )];

        let (overall, metrics, divergent) = compute_metrics(&judgements);

        assert!(overall < 100.0);
        assert_eq!(divergent.len(), 1); // 0.151 > 0.15, so flagged
        assert!(divergent[0].max_disagreement > 0.15);
    }

    #[test]
    fn test_metrics_multiple_rubrics() {
        // Multiple different rubrics with different agreement levels
        let judgements = vec![
            (
                "test1".to_string(),
                "rubric_A".to_string(),
                vec![0.9, 0.9, 0.9],
                vec!["j1".to_string(), "j2".to_string(), "j3".to_string()],
            ),
            (
                "test2".to_string(),
                "rubric_B".to_string(),
                vec![0.5, 0.9], // Divergent
                vec!["j1".to_string(), "j2".to_string()],
            ),
        ];

        let (overall, metrics, divergent) = compute_metrics(&judgements);

        assert_eq!(metrics.len(), 2);
        assert_eq!(metrics["rubric_A"].agreement_percentage, 100.0);
        assert!(metrics["rubric_B"].agreement_percentage < 100.0);
        assert_eq!(divergent.len(), 1);
        assert_eq!(divergent[0].rubric, "rubric_B");
    }

    #[test]
    fn test_metrics_extreme_score_ranges() {
        // Judges give extreme ranges: 0.0 to 1.0
        let judgements = vec![(
            "test1".to_string(),
            "rubric1".to_string(),
            vec![0.0, 0.5, 1.0],
            vec!["j1".to_string(), "j2".to_string(), "j3".to_string()],
        )];

        let (overall, metrics, divergent) = compute_metrics(&judgements);

        // Max disagreement = 1.0 - 0.0 = 1.0, clearly divergent
        assert_eq!(divergent.len(), 1);
        assert_eq!(divergent[0].max_disagreement, 1.0);
        assert_eq!(divergent[0].judge_scores["j1"], 0.0);
        assert_eq!(divergent[0].judge_scores["j3"], 1.0);
    }

    #[test]
    fn test_metrics_near_boundary_scores() {
        // All scores very close but within bounds
        let judgements = vec![(
            "test1".to_string(),
            "rubric1".to_string(),
            vec![0.80, 0.81, 0.82],
            vec!["j1".to_string(), "j2".to_string(), "j3".to_string()],
        )];

        let (overall, metrics, divergent) = compute_metrics(&judgements);

        assert!(overall > 95.0); // Very high agreement
        assert_eq!(divergent.len(), 0); // 0.02 < 0.15
    }

    // ── Regression Tracking Tests ───────────────────────────────────────────

    #[test]
    fn test_regression_no_change() {
        let current = JudgeValidationReport {
            suite_name: "test".to_string(),
            total_tests: 5,
            tests_with_judges: 3,
            judges_compared: vec!["j1".to_string(), "j2".to_string()],
            overall_agreement: 90.0,
            per_rubric_metrics: HashMap::new(),
            divergent_cases: vec![],
            summary: ValidationSummary {
                is_reliable: true,
                confidence_level: "high".to_string(),
                recommendations: vec![],
            },
            regression_data: None,
        };

        let baseline = JudgeValidationBaseline {
            timestamp: "2026-07-21".to_string(),
            suite_name: "test".to_string(),
            overall_agreement: 90.0,
            per_judge_agreement: [("j1".to_string(), 90.0), ("j2".to_string(), 90.0)]
                .iter()
                .cloned()
                .collect(),
            per_rubric_metrics: HashMap::new(),
        };

        let regression = compute_regression(&current, &baseline);

        assert_eq!(regression.agreement_delta, 0.0);
        assert!(!regression.is_regression);
        assert_eq!(regression.trend_direction, TrendDirection::Stable);
    }

    #[test]
    fn test_regression_improvement() {
        let current = JudgeValidationReport {
            suite_name: "test".to_string(),
            total_tests: 5,
            tests_with_judges: 3,
            judges_compared: vec!["j1".to_string()],
            overall_agreement: 95.0,
            per_rubric_metrics: HashMap::new(),
            divergent_cases: vec![],
            summary: ValidationSummary {
                is_reliable: true,
                confidence_level: "high".to_string(),
                recommendations: vec![],
            },
            regression_data: None,
        };

        let baseline = JudgeValidationBaseline {
            timestamp: "2026-07-20".to_string(),
            suite_name: "test".to_string(),
            overall_agreement: 90.0,
            per_judge_agreement: [("j1".to_string(), 90.0)].iter().cloned().collect(),
            per_rubric_metrics: HashMap::new(),
        };

        let regression = compute_regression(&current, &baseline);

        assert_eq!(regression.agreement_delta, 5.0); // +5%
        assert!(!regression.is_regression); // Positive delta
        assert_eq!(regression.trend_direction, TrendDirection::Improving);
    }

    #[test]
    fn test_regression_minor_degradation() {
        let current = JudgeValidationReport {
            suite_name: "test".to_string(),
            total_tests: 5,
            tests_with_judges: 3,
            judges_compared: vec!["j1".to_string()],
            overall_agreement: 88.0,
            per_rubric_metrics: HashMap::new(),
            divergent_cases: vec![],
            summary: ValidationSummary {
                is_reliable: true,
                confidence_level: "high".to_string(),
                recommendations: vec![],
            },
            regression_data: None,
        };

        let baseline = JudgeValidationBaseline {
            timestamp: "2026-07-20".to_string(),
            suite_name: "test".to_string(),
            overall_agreement: 90.0,
            per_judge_agreement: [("j1".to_string(), 90.0)].iter().cloned().collect(),
            per_rubric_metrics: HashMap::new(),
        };

        let regression = compute_regression(&current, &baseline);

        assert_eq!(regression.agreement_delta, -2.0); // -2%
        assert!(!regression.is_regression); // -2% is within tolerance
        assert_eq!(regression.trend_direction, TrendDirection::Stable); // Within ±2%
    }

    #[test]
    fn test_regression_major_degradation() {
        let current = JudgeValidationReport {
            suite_name: "test".to_string(),
            total_tests: 5,
            tests_with_judges: 3,
            judges_compared: vec!["j1".to_string()],
            overall_agreement: 84.0,
            per_rubric_metrics: HashMap::new(),
            divergent_cases: vec![],
            summary: ValidationSummary {
                is_reliable: false,
                confidence_level: "low".to_string(),
                recommendations: vec!["Investigate judge degradation".to_string()],
            },
            regression_data: None,
        };

        let baseline = JudgeValidationBaseline {
            timestamp: "2026-07-20".to_string(),
            suite_name: "test".to_string(),
            overall_agreement: 90.0,
            per_judge_agreement: [("j1".to_string(), 90.0)].iter().cloned().collect(),
            per_rubric_metrics: HashMap::new(),
        };

        let regression = compute_regression(&current, &baseline);

        assert_eq!(regression.agreement_delta, -6.0); // -6%
        assert!(regression.is_regression); // Exceeds -5% threshold
        assert_eq!(regression.trend_direction, TrendDirection::Degrading);
    }

    #[test]
    fn test_regression_judge_degradation() {
        let current = JudgeValidationReport {
            suite_name: "test".to_string(),
            total_tests: 5,
            tests_with_judges: 3,
            judges_compared: vec!["j1".to_string(), "j2".to_string()],
            overall_agreement: 85.0,
            per_rubric_metrics: HashMap::new(),
            divergent_cases: vec![],
            summary: ValidationSummary {
                is_reliable: false,
                confidence_level: "medium".to_string(),
                recommendations: vec![],
            },
            regression_data: None,
        };

        let baseline = JudgeValidationBaseline {
            timestamp: "2026-07-20".to_string(),
            suite_name: "test".to_string(),
            overall_agreement: 92.0,
            per_judge_agreement: [("j1".to_string(), 92.0), ("j2".to_string(), 92.0)]
                .iter()
                .cloned()
                .collect(),
            per_rubric_metrics: HashMap::new(),
        };

        let regression = compute_regression(&current, &baseline);

        assert_eq!(regression.judge_degradation.len(), 2);
        assert!(regression.judge_degradation.contains_key("j1"));
        assert!(regression.judge_degradation.contains_key("j2"));
        assert_eq!(
            regression.judge_degradation["j1"].degradation_rate,
            7.0 // 92 - 85
        );
    }

    #[test]
    fn test_trend_direction_boundary() {
        // Exactly at -2% boundary (should be Stable, not Degrading)
        let current = JudgeValidationReport {
            suite_name: "test".to_string(),
            total_tests: 5,
            tests_with_judges: 3,
            judges_compared: vec!["j1".to_string()],
            overall_agreement: 88.0,
            per_rubric_metrics: HashMap::new(),
            divergent_cases: vec![],
            summary: ValidationSummary {
                is_reliable: true,
                confidence_level: "high".to_string(),
                recommendations: vec![],
            },
            regression_data: None,
        };

        let baseline = JudgeValidationBaseline {
            timestamp: "2026-07-20".to_string(),
            suite_name: "test".to_string(),
            overall_agreement: 90.0,
            per_judge_agreement: [("j1".to_string(), 90.0)].iter().cloned().collect(),
            per_rubric_metrics: HashMap::new(),
        };

        let regression = compute_regression(&current, &baseline);
        assert_eq!(regression.trend_direction, TrendDirection::Stable);
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
    fallback_config: FallbackConfig,
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
    let outcome =
        runner::execute_suite(suite_path, None, &run_args, &HashMap::new(), &embed_client).await?;

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
        println!("  {} No llm_judge assertions found in suite", "ℹ".cyan());
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
            regression_data: None,
        });
    }

    println!(
        "  {} Found {} judge assertions",
        "✓".green(),
        test_judge_pairs.len()
    );
    println!();

    // ── Phase 3: Collect judge scores for each output ──
    println!("  {} Collecting judge scores...", "▸".cyan());
    let pb = ProgressBar::new(test_judge_pairs.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.cyan} [{bar:20.cyan/blue}] {pos}/{len}")
            .unwrap(),
    );

    let mut judgements: Vec<(String, String, Vec<f64>, Vec<String>)> = Vec::new();
    let judge_names: Vec<String> = judge_refs.iter().map(|j| j.model.clone()).collect();

    for (test_name, output, rubric) in test_judge_pairs {
        pb.inc(1);

        let mut scores = Vec::new();
        let mut has_errors = false;

        // Run output through each judge
        for judge in &judge_refs {
            match score_output_with_judge(judge, &output, &rubric).await {
                Ok(score) => {
                    scores.push(score);
                }
                Err(e) => {
                    has_errors = true;
                    let error_msg = e.to_string();

                    // Determine fallback score based on configured strategy
                    if let Some(fallback_score) = fallback_config.get_fallback_score(&error_msg) {
                        // Log the error with appropriate level
                        if error_msg.contains("401") || error_msg.contains("Unauthorized") {
                            eprintln!("  {} {}: Invalid API key (401)", "✗".red(), judge.model);
                        } else if error_msg.contains("timeout") || error_msg.contains("timed out") {
                            eprintln!("  {} {}: Timeout", "⏱".yellow(), judge.model);
                        } else if error_msg.contains("unparseable") {
                            eprintln!("  {} {}: Bad JSON response", "⚠".yellow(), judge.model);
                        } else {
                            eprintln!("  {} {}: {}", "⚠".yellow(), judge.model, error_msg);
                        }
                        scores.push(fallback_score);
                    } else {
                        // Skip strategy or will compute average later
                        eprintln!(
                            "  {} {}: Skipping due to strategy ({})",
                            "⊘".yellow(),
                            judge.model,
                            format!("{:?}", fallback_config.strategy).to_lowercase()
                        );
                    }
                }
            }
        }

        judgements.push((test_name, rubric, scores, judge_names.clone()));

        if has_errors && judgements.len().is_multiple_of(5) {
            pb.println(format!(
                "  {} Some judges failed; using fallback scores",
                "⚠".yellow()
            ));
        }
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
        regression_data: None,
    })
}
