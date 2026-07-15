use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use colored::Colorize;

use crate::config::{self, Assertion};
use crate::providers::ModelRef;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct JudgeValidationReport {
    pub total_tests: usize,
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
    pub avg_confidence: f64,
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
    pub confidence_level: String, // "high", "medium", "low"
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone)]
struct JudgeDecision {
    judge_name: String,
    score: f64,
    #[allow(dead_code)]
    passed: bool,
    #[allow(dead_code)]
    reason: String,
}

pub async fn validate_judge(
    suite_paths: Vec<String>,
    judges: Vec<String>,
) -> Result<JudgeValidationReport> {
    println!("\n{} Judge Validation", "🔍".cyan().bold());
    println!("{}", "─".repeat(70).dimmed());
    println!("  Suites:   {}", suite_paths.join(", ").yellow());
    println!("  Judges:   {}", judges.join(", ").yellow());
    println!();

    let mut all_rubric_decisions: HashMap<String, Vec<Vec<JudgeDecision>>> = HashMap::new();
    let mut all_divergent_cases: Vec<DivergentCase> = Vec::new();
    let mut total_test_count = 0;

    // Process each suite
    for suite_path in suite_paths {
        let suite_content = std::fs::read_to_string(&suite_path)?;
        let suite: config::Suite = toml::from_str(&suite_content)?;

        println!("  {} {}", "▸".cyan(), suite_path.yellow());

        // Extract tests with llm_judge assertions
        for test in &suite.tests {
            let llm_judge_assertions: Vec<_> = test
                .assert
                .iter()
                .filter_map(|a| match a {
                    Assertion::LlmJudge { rubric, threshold, .. } => {
                        Some((rubric.clone(), *threshold))
                    }
                    _ => None,
                })
                .collect();

            if llm_judge_assertions.is_empty() {
                continue;
            }

            total_test_count += 1;

            // For each rubric in this test, collect judge decisions
            for (rubric, threshold) in llm_judge_assertions {
                let mut decisions = Vec::new();

                for judge_spec in &judges {
                    let judge = ModelRef::resolve(judge_spec, "http://localhost:11434")?;
                    match judge.provider.chat(&judge.model, None, &rubric, 0.0).await {
                        Ok(_result) => {
                            // In a real scenario, we'd parse the response
                            // For now, use a placeholder
                            decisions.push(JudgeDecision {
                                judge_name: judge_spec.clone(),
                                score: 0.85,
                                passed: 0.85 >= threshold,
                                reason: "placeholder".to_string(),
                            });
                        }
                        Err(e) => {
                            eprintln!("  {} Failed to run judge {}: {}", "⚠".yellow(), judge_spec, e);
                        }
                    }
                }

                all_rubric_decisions
                    .entry(rubric.clone())
                    .or_insert_with(Vec::new)
                    .push(decisions.clone());

                // Check for divergence
                if decisions.len() > 1 {
                    let max_score = decisions.iter().map(|d| d.score).fold(f64::NEG_INFINITY, f64::max);
                    let min_score = decisions.iter().map(|d| d.score).fold(f64::INFINITY, f64::min);
                    let disagreement = max_score - min_score;

                    if disagreement > 0.15 {
                        let mut judge_scores = HashMap::new();
                        for decision in &decisions {
                            judge_scores.insert(decision.judge_name.clone(), decision.score);
                        }

                        all_divergent_cases.push(DivergentCase {
                            test_name: test.name.clone(),
                            rubric: rubric.clone(),
                            judge_scores,
                            max_disagreement: disagreement,
                        });
                    }
                }
            }
        }
    }

    // Compute agreement metrics
    let mut per_rubric_metrics = HashMap::new();
    for (rubric, all_decisions) in all_rubric_decisions {
        let mut total_agreement = 0.0;
        let mut disagreement_count = 0;

        for decisions in &all_decisions {
            if decisions.len() > 1 {
                let max_score = decisions.iter().map(|d| d.score).fold(f64::NEG_INFINITY, f64::max);
                let min_score = decisions.iter().map(|d| d.score).fold(f64::INFINITY, f64::min);
                let agreement = 1.0 - (max_score - min_score).min(1.0);
                total_agreement += agreement;

                if (max_score - min_score) > 0.15 {
                    disagreement_count += 1;
                }
            }
        }

        let agreement_pct = if all_decisions.is_empty() {
            100.0
        } else {
            (total_agreement / all_decisions.len() as f64) * 100.0
        };

        per_rubric_metrics.insert(
            rubric.clone(),
            RubricMetrics {
                rubric_name: rubric,
                test_count: all_decisions.len(),
                agreement_percentage: agreement_pct,
                avg_confidence: 0.85, // placeholder
                disagreement_count,
            },
        );
    }

    let overall_agreement = if per_rubric_metrics.is_empty() {
        100.0
    } else {
        per_rubric_metrics
            .values()
            .map(|m| m.agreement_percentage)
            .sum::<f64>()
            / per_rubric_metrics.len() as f64
    };

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
    if all_divergent_cases.len() > (total_test_count / 10).max(1) {
        recommendations.push(format!(
            "High divergence rate ({} cases) — consider rubric clarification",
            all_divergent_cases.len()
        ));
    }

    Ok(JudgeValidationReport {
        total_tests: total_test_count,
        judges_compared: judges,
        overall_agreement,
        per_rubric_metrics,
        divergent_cases: all_divergent_cases,
        summary: ValidationSummary {
            is_reliable,
            confidence_level,
            recommendations,
        },
    })
}
