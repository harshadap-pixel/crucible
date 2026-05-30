use anyhow::Result;
use colored::Colorize;
use comfy_table::{Cell, ContentArrangement, Table};

use crate::cli::ReportArgs;
use crate::runner::TestResult;
use crate::store::{db::TestRecord, Db};

/// Print a single run report to the terminal
pub fn print_run(
    run_id: &str,
    timestamp: &str,
    model: &str,
    _suite_name: &str,
    results: &[TestResult],
    baseline: Option<&[TestRecord]>,
    reg_thresh: f64,
) {
    println!();
    println!(
        "{}  {}  model: {}",
        format!("Run #{run_id}").bold(),
        timestamp.dimmed(),
        model.yellow()
    );
    println!("{}", "─".repeat(70).dimmed());

    let mut table = Table::new();
    table.load_preset(comfy_table::presets::NOTHING);
    table.set_content_arrangement(ContentArrangement::Dynamic);
    // Only show TTFT column when at least one result has a non-zero value
    let show_ttft = results.iter().any(|r| r.ttft_ms > 0);

    let mut headers = vec![
        Cell::new("TEST").add_attribute(comfy_table::Attribute::Bold),
        Cell::new("SCORE").add_attribute(comfy_table::Attribute::Bold),
        Cell::new("STATUS").add_attribute(comfy_table::Attribute::Bold),
        Cell::new("LATENCY").add_attribute(comfy_table::Attribute::Bold),
    ];
    if show_ttft {
        headers.push(Cell::new("TTFT").add_attribute(comfy_table::Attribute::Bold));
    }
    headers.push(Cell::new("DELTA").add_attribute(comfy_table::Attribute::Bold));
    headers.push(Cell::new("FLAG").add_attribute(comfy_table::Attribute::Bold));
    table.set_header(headers);

    let mut regression_count = 0usize;

    for r in results {
        // Find baseline score for this test
        let baseline_record =
            baseline.and_then(|bs| bs.iter().find(|b| b.test_name == r.test_name));

        let (delta_str, flag_str) = if let Some(base) = baseline_record {
            let delta = r.score - base.score;
            let flip = base.passed && !r.passed;
            let drift = delta < -reg_thresh;

            let ds = format!("{:+.3}", delta);
            let ds = if delta >= 0.0 {
                ds.green().to_string()
            } else {
                ds.red().to_string()
            };

            let flag = match (flip, drift) {
                (true, true) => {
                    regression_count += 1;
                    "⚠ REGRESSION".red().bold().to_string()
                }
                (true, false) => {
                    regression_count += 1;
                    "⚠ FLIP".red().to_string()
                }
                (false, true) => {
                    regression_count += 1;
                    "⚠ DRIFT".yellow().to_string()
                }
                _ => String::new(),
            };
            (ds, flag)
        } else {
            ("—".dimmed().to_string(), String::new())
        };

        let status = if r.passed {
            "✅ PASS".green().to_string()
        } else {
            "❌ FAIL".red().to_string()
        };

        let mut row = vec![
            Cell::new(&r.test_name),
            Cell::new(format!("{:.3}", r.score)),
            Cell::new(status),
            Cell::new(format!("{}ms", r.latency_ms)),
        ];
        if show_ttft {
            let ttft_cell = if r.ttft_ms > 0 {
                Cell::new(format!("{}ms", r.ttft_ms))
            } else {
                Cell::new("—")
            };
            row.push(ttft_cell);
        }
        row.push(Cell::new(delta_str));
        row.push(Cell::new(flag_str));
        table.add_row(row);
    }

    println!("{table}");
    println!("{}", "─".repeat(70).dimmed());

    let total = results.len();
    let passed = results.iter().filter(|r| r.passed).count();
    let avg = if total > 0 {
        results.iter().map(|r| r.score).sum::<f64>() / total as f64
    } else {
        0.0
    };

    let summary_line = format!(
        "SUMMARY   {}/{} passed   avg score: {:.3}",
        passed, total, avg
    );

    if regression_count > 0 {
        println!(
            "  {}   {} regression(s)",
            summary_line.bold(),
            regression_count.to_string().red().bold()
        );
    } else if baseline.is_some() {
        println!("  {}   {}", summary_line.bold(), "✓ no regressions".green());
    } else {
        println!("  {}", summary_line.bold());
    }
}

/// `crucible report` subcommand — show run history table
pub async fn history(args: ReportArgs) -> Result<()> {
    let db = Db::open()?;
    let runs = db.recent_runs(args.last, args.suite.as_deref())?;

    if runs.is_empty() {
        println!("  No runs found. Use `crucible run` to start evaluating.");
        return Ok(());
    }

    println!("{}", "\nRUN HISTORY".bold().cyan());
    println!("{}", "─".repeat(70).dimmed());

    let mut table = Table::new();
    table.load_preset(comfy_table::presets::NOTHING);
    table.set_header(vec![
        "RUN ID",
        "SUITE",
        "MODEL",
        "TIMESTAMP",
        "PASS",
        "AVG SCORE",
        "",
    ]);

    for r in &runs {
        let baseline_marker = if r.is_baseline {
            "● baseline".yellow().to_string()
        } else {
            String::new()
        };
        table.add_row(vec![
            r.id.clone(),
            r.suite_name.clone(),
            r.model.clone(),
            r.timestamp.clone(),
            format!("{}/{}", r.passed_tests, r.total_tests),
            format!("{:.3}", r.avg_score),
            baseline_marker,
        ]);
    }

    println!("{table}");
    println!();
    Ok(())
}
