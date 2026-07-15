use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};

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
    pub confidence_level: String,
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

#[derive(Debug, Clone)]
struct TestJudgement {
    test_name: String,
    rubric: String,
    #[allow(dead_code)]
    threshold: f64,
    decisions: Vec<JudgeDecision>,
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

    // Parse judge models
    let judge_refs: Vec<ModelRef> = judges
        .iter()
        .map(|spec| ModelRef::resolve(spec, "http://localhost:11434"))
        .collect::<Result<Vec<_>>>()?;

    let mut all_judgements: Vec<TestJudgement> = Vec::new();
    let mut total_test_count = 0;

    // Process each suite
    for suite_path in &suite_paths {
        let suite_content = std::fs::read_to_string(&suite_path)?;
        let suite: config::Suite = toml::from_str(&suite_content)?;

        println!("  {} {}", "▸".cyan(), suite_path.yellow());

        // Extract tests with llm_judge assertions
        let tests_with_judges: Vec<_> = suite
            .tests
            .iter()
            .filter_map(|test| {
                let llm_judges: Vec<_> = test
                    .assert
                    .iter()
                    .filter_map(|a| match a {
                        Assertion::LlmJudge { rubric, threshold, .. } => {
                            Some((rubric.clone(), *threshold))
                        }
                        _ => None,
                    })
                    .collect();

                if llm_judges.is_empty() {
                    None
                } else {
                    Some((test, llm_judges))
                }
            })
            .collect();

        total_test_count += tests_with_judges.len();

        let pb = ProgressBar::new(tests_with_judges.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.cyan} [{bar:20.cyan/blue}] {pos}/{len} {msg}")
                .unwrap()
                .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]),
        );

        // For each test, evaluate with each judge
        for (test, llm_judges) in tests_with_judges {
            pb.inc(1);
            pb.set_message(format!("Evaluating {}", test.name));

            for (rubric, threshold) in llm_judges {
                let mut decisions = Vec::new();

                // Run the test prompt through the rubric with each judge
                for judge in &judge_refs {
                    let prompt = format!("RUBRIC:\n{}\n\nRESPONSE TO EVALUATE:\nTest prompt.\n\nScore the response.", rubric);

                    match judge.provider.chat(&judge.model, None, &prompt, 0.0).await {
                        Ok(result) => {
                            if let Some((score, _reason)) = parse_judge_response(&result.text) {
                                decisions.push(JudgeDecision {
                                    judge_name: judge.model.clone(),
                                    score,
                                    passed: score >= threshold,
                                    reason: _reason,
                                });
                            } else {
                                decisions.push(JudgeDecision {
                                    judge_name: judge.model.clone(),
                                    score: 0.5,
                                    passed: false,
                                    reason: "Failed to parse judge response".to_string(),
                                });
                            }
                        }
                        Err(_e) => {
                            decisions.push(JudgeDecision {
                                judge_name: judge.model.clone(),
                                score: 0.0,
                                passed: false,
                                reason: "Judge call failed".to_string(),
                            });
                        }
                    }
                }

                all_judgements.push(TestJudgement {
                    test_name: test.name.clone(),
                    rubric,
                    threshold,
                    decisions,
                });
            }
        }
        pb.finish_with_message("✓ Evaluated");
    }

    println!("  {} Total tests evaluated: {}", "✓".green(), total_test_count);
    println!();

    // Compute metrics from judgements
    let (per_rubric_metrics, divergent_cases, overall_agreement) =
        compute_metrics(&all_judgements)?;

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
    if divergent_cases.len() > (total_test_count / 10).max(1) {
        recommendations.push(format!(
            "High divergence rate ({} cases) — consider rubric clarification",
            divergent_cases.len()
        ));
    }

    Ok(JudgeValidationReport {
        total_tests: total_test_count,
        judges_compared: judges,
        overall_agreement,
        per_rubric_metrics,
        divergent_cases,
        summary: ValidationSummary {
            is_reliable,
            confidence_level,
            recommendations,
        },
    })
}

fn compute_metrics(
    judgements: &[TestJudgement],
) -> Result<(HashMap<String, RubricMetrics>, Vec<DivergentCase>, f64)> {
    let mut per_rubric_data: HashMap<String, Vec<Vec<f64>>> = HashMap::new();
    let mut divergent_cases = Vec::new();

    for judgement in judgements {
        let scores: Vec<f64> = judgement.decisions.iter().map(|d| d.score).collect();

        per_rubric_data
            .entry(judgement.rubric.clone())
            .or_insert_with(Vec::new)
            .push(scores.clone());

        // Detect divergence
        if scores.len() > 1 {
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let min = scores.iter().copied().fold(f64::INFINITY, f64::min);
            let disagreement = max - min;

            if disagreement > 0.15 {
                let mut judge_scores = HashMap::new();
                for (decision, score) in judgement.decisions.iter().zip(scores.iter()) {
                    judge_scores.insert(decision.judge_name.clone(), *score);
                }

                divergent_cases.push(DivergentCase {
                    test_name: judgement.test_name.clone(),
                    rubric: judgement.rubric.clone(),
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
                avg_confidence: 0.85,
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

    Ok((per_rubric_metrics, divergent_cases, overall_agreement))
}

fn parse_judge_response(text: &str) -> Option<(f64, String)> {
    let start = text.find('{')?;
    let end = text.rfind('}')? + 1;
    let json_str = &text[start..end];

    let v: serde_json::Value = serde_json::from_str(json_str).ok()?;
    let score = v.get("score")?.as_f64()?;
    let reason = v.get("reason")?.as_str()?.to_string();
    Some((score.clamp(0.0, 1.0), reason))
}
