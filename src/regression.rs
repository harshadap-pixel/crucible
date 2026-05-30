use crate::cli::CompareArgs;
use crate::store::{db::TestRecord, Db};
use anyhow::Result;
use colored::Colorize;

#[derive(Debug)]
pub enum RegressionKind {
    ScoreDrift {
        #[allow(dead_code)]
        delta: f64,
    },
    PassFailFlip,
    Both {
        #[allow(dead_code)]
        delta: f64,
    },
}

#[derive(Debug)]
pub struct RegressionResult {
    pub test_name: String,
    pub baseline_score: f64,
    pub current_score: f64,
    #[allow(dead_code)]
    pub baseline_passed: bool,
    #[allow(dead_code)]
    pub current_passed: bool,
    pub kind: Option<RegressionKind>,
}

pub fn detect(
    baseline: &[TestRecord],
    current: &[TestRecord],
    threshold: f64,
) -> Vec<RegressionResult> {
    current
        .iter()
        .filter_map(|curr| {
            let base = baseline.iter().find(|b| b.test_name == curr.test_name)?;

            let score_delta = curr.score - base.score;
            let flipped = base.passed && !curr.passed;
            let drifted = score_delta < -threshold;

            let kind = match (flipped, drifted) {
                (true, true) => Some(RegressionKind::Both { delta: score_delta }),
                (true, false) => Some(RegressionKind::PassFailFlip),
                (false, true) => Some(RegressionKind::ScoreDrift { delta: score_delta }),
                (false, false) => None,
            };

            Some(RegressionResult {
                test_name: curr.test_name.clone(),
                baseline_score: base.score,
                current_score: curr.score,
                baseline_passed: base.passed,
                current_passed: curr.passed,
                kind,
            })
        })
        .collect()
}

/// `crucible compare <run_a> <run_b>` subcommand
pub async fn compare(args: CompareArgs) -> Result<()> {
    let db = Db::open()?;

    let base_results = db.get_results_for_run(&args.run_a)?;
    let curr_results = db.get_results_for_run(&args.run_b)?;

    if base_results.is_empty() {
        anyhow::bail!("No results for run '{}'", args.run_a);
    }
    if curr_results.is_empty() {
        anyhow::bail!("No results for run '{}'", args.run_b);
    }

    let regressions = detect(&base_results, &curr_results, 0.05);

    println!(
        "\n{} {} → {}",
        "COMPARE".bold().cyan(),
        args.run_a.yellow(),
        args.run_b.yellow()
    );
    println!("{}", "─".repeat(62).dimmed());

    let reg_count = regressions.iter().filter(|r| r.kind.is_some()).count();

    for r in &regressions {
        let delta_str = format!("{:+.3}", r.current_score - r.baseline_score);
        let delta_col = if r.current_score >= r.baseline_score {
            delta_str.green().to_string()
        } else {
            delta_str.red().to_string()
        };

        let flag = match &r.kind {
            Some(RegressionKind::Both { .. }) => {
                "⚠ REGRESSION (flip+drift)".red().bold().to_string()
            }
            Some(RegressionKind::PassFailFlip) => "⚠ REGRESSION (flip)".red().to_string(),
            Some(RegressionKind::ScoreDrift { .. }) => "⚠ REGRESSION (drift)".yellow().to_string(),
            None => String::new(),
        };

        println!(
            "  {:<35} {:.3} → {:.3}  {}  {}",
            r.test_name, r.baseline_score, r.current_score, delta_col, flag
        );
    }

    println!("{}", "─".repeat(62).dimmed());
    if reg_count == 0 {
        println!("  {} No regressions detected", "✓".green());
    } else {
        println!("  {} {} regression(s) detected", "⚠".yellow(), reg_count);
    }
    println!();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::db::TestRecord;

    fn rec(name: &str, score: f64, passed: bool) -> TestRecord {
        TestRecord {
            run_id: "run".into(),
            test_name: name.into(),
            score,
            passed,
            pass_rate: 1.0,
            latency_ms: 0,
            ttft_ms: 0,
            input_tokens: 0,
            output_tokens: 0,
            reason: String::new(),
        }
    }

    #[test]
    fn no_regression_when_scores_equal() {
        let base = vec![rec("t1", 1.0, true)];
        let curr = vec![rec("t1", 1.0, true)];
        let results = detect(&base, &curr, 0.05);
        assert_eq!(results.len(), 1);
        assert!(results[0].kind.is_none(), "should have no regression");
    }

    #[test]
    fn detects_score_drift() {
        let base = vec![rec("t1", 1.0, true)];
        let curr = vec![rec("t1", 0.8, true)]; // -0.2 > threshold 0.05
        let results = detect(&base, &curr, 0.05);
        assert!(matches!(
            results[0].kind,
            Some(RegressionKind::ScoreDrift { .. })
        ));
    }

    #[test]
    fn detects_pass_fail_flip() {
        let base = vec![rec("t1", 1.0, true)];
        let curr = vec![rec("t1", 1.0, false)]; // flipped, no score drift
        let results = detect(&base, &curr, 0.05);
        assert!(matches!(
            results[0].kind,
            Some(RegressionKind::PassFailFlip)
        ));
    }

    #[test]
    fn detects_both_flip_and_drift() {
        let base = vec![rec("t1", 1.0, true)];
        let curr = vec![rec("t1", 0.5, false)]; // flipped + big drift
        let results = detect(&base, &curr, 0.05);
        assert!(matches!(results[0].kind, Some(RegressionKind::Both { .. })));
    }

    #[test]
    fn ignores_tests_not_in_baseline() {
        let base = vec![rec("t1", 1.0, true)];
        let curr = vec![
            rec("t1", 1.0, true),
            rec("t2", 0.0, false), // new test — no baseline to compare
        ];
        let results = detect(&base, &curr, 0.05);
        // t2 filtered out (no baseline match); only t1 present
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].test_name, "t1");
    }

    #[test]
    fn improvement_is_not_flagged() {
        let base = vec![rec("t1", 0.7, true)];
        let curr = vec![rec("t1", 0.95, true)]; // improved
        let results = detect(&base, &curr, 0.05);
        assert!(results[0].kind.is_none());
    }
}
