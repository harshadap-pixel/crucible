/// Dataset evaluation — load JSONL/CSV rows, expand into TestCases via a
/// template suite, run through the standard runner, and report aggregate
/// statistics with features not found in any other local eval tool.
use anyhow::{bail, Context, Result};
use colored::Colorize;
use std::collections::HashMap;

use crate::config::{Assertion, TestCase};

// ── Public types ──────────────────────────────────────────────────────────────

/// One row from the dataset file.
pub type Row = HashMap<String, String>;

/// Aggregate statistics over all dataset rows.
#[derive(Debug)]
pub struct DatasetStats {
    pub total: usize,
    pub passed: usize,
    pub avg_score: f64,
    pub p10: f64,
    pub p50: f64,
    pub p90: f64,
    pub ttft_p50_ms: u64,
    pub ttft_p90_ms: u64,
    /// (slice_value, passed, total, avg_score) — populated when --slice-by is set
    pub slices: Vec<(String, usize, usize, f64)>,
    /// Worst rows: (row_index, score, input_snippet, output_snippet, expected_snippet)
    pub failures: Vec<RowFailure>,
    /// Score histogram buckets [0,5) over [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    pub histogram: [usize; 5],
}

#[derive(Debug)]
pub struct RowFailure {
    pub index: usize,
    pub score: f64,
    pub input: String,
    pub output: String,
    pub expected: String,
    pub reason: String,
}

// ── File loading ──────────────────────────────────────────────────────────────

/// Load a dataset from a JSONL or CSV file.
/// Format is auto-detected from the file extension (.csv → CSV, else JSONL).
pub fn load(path: &str) -> Result<Vec<Row>> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("Cannot read dataset file: {path}"))?;

    if path.ends_with(".csv") {
        load_csv(&content)
    } else {
        load_jsonl(&content, path)
    }
}

fn load_jsonl(content: &str, path: &str) -> Result<Vec<Row>> {
    let mut rows = Vec::new();
    for (i, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with("//") {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(line)
            .with_context(|| format!("{path}:{} — invalid JSON: {line}", i + 1))?;

        let obj = v
            .as_object()
            .with_context(|| format!("{path}:{} — expected a JSON object", i + 1))?;

        let mut row: Row = HashMap::new();
        for (k, v) in obj {
            // Flatten all scalar values to strings; arrays become comma-joined
            let s = match v {
                serde_json::Value::String(s) => s.clone(),
                serde_json::Value::Number(n) => n.to_string(),
                serde_json::Value::Bool(b) => b.to_string(),
                serde_json::Value::Array(arr) => arr
                    .iter()
                    .map(|x| x.as_str().unwrap_or("").to_string())
                    .collect::<Vec<_>>()
                    .join(", "),
                _ => v.to_string(),
            };
            row.insert(k.clone(), s);
        }
        rows.push(row);
    }
    if rows.is_empty() {
        bail!("Dataset file '{path}' contains no rows");
    }
    Ok(rows)
}

fn load_csv(content: &str) -> Result<Vec<Row>> {
    let mut lines = content.lines();
    let header_line = lines
        .next()
        .ok_or_else(|| anyhow::anyhow!("CSV is empty"))?;
    let headers: Vec<&str> = header_line.split(',').map(|h| h.trim()).collect();

    let mut rows = Vec::new();
    for line in lines {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        // Simple CSV split — handles quoted fields with commas
        let values = split_csv_line(line);
        let mut row: Row = HashMap::new();
        for (i, h) in headers.iter().enumerate() {
            let val = values
                .get(i)
                .map(|s| s.trim_matches('"').to_string())
                .unwrap_or_default();
            row.insert(h.to_string(), val);
        }
        rows.push(row);
    }
    if rows.is_empty() {
        bail!("CSV file contains no data rows");
    }
    Ok(rows)
}

/// Naive CSV line splitter that handles double-quoted fields.
fn split_csv_line(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut in_quotes = false;
    for ch in line.chars() {
        match ch {
            '"' => in_quotes = !in_quotes,
            ',' if !in_quotes => {
                out.push(cur.clone());
                cur.clear();
            }
            _ => cur.push(ch),
        }
    }
    out.push(cur);
    out
}

// ── MMLU auto-detection ───────────────────────────────────────────────────────

/// Returns true if the rows look like MMLU format.
/// MMLU has: question, A, B, C, D, answer (letter or index)
pub fn is_mmlu(rows: &[Row]) -> bool {
    if rows.is_empty() {
        return false;
    }
    let r = &rows[0];
    (r.contains_key("question") || r.contains_key("Question"))
        && (r.contains_key("A") || r.contains_key("choices"))
        && r.contains_key("answer")
}

/// Convert MMLU rows into normalised `input` / `expected` / `choices` rows
/// so the standard template expansion works.
pub fn normalise_mmlu(rows: Vec<Row>) -> Vec<Row> {
    rows.into_iter()
        .map(|mut r| {
            let question = r
                .get("question")
                .or_else(|| r.get("Question"))
                .cloned()
                .unwrap_or_default();

            let a = r.get("A").cloned().unwrap_or_default();
            let b = r.get("B").cloned().unwrap_or_default();
            let c = r.get("C").cloned().unwrap_or_default();
            let d = r.get("D").cloned().unwrap_or_default();

            let answer_raw = r.get("answer").cloned().unwrap_or_default();
            // MMLU answer can be a letter (A/B/C/D) or a 0-based index
            let answer_letter = match answer_raw.trim() {
                "0" => "A",
                "1" => "B",
                "2" => "C",
                "3" => "D",
                l => l,
            };
            let expected_text = match answer_letter {
                "A" => a.clone(),
                "B" => b.clone(),
                "C" => c.clone(),
                "D" => d.clone(),
                _ => answer_letter.to_string(),
            };

            let input = format!(
                "{question}\nA) {a}\nB) {b}\nC) {c}\nD) {d}\n\nAnswer with only the letter (A, B, C, or D)."
            );

            r.insert("input".into(), input);
            r.insert("expected".into(), expected_text);
            r.insert("answer_letter".into(), answer_letter.to_string());
            r
        })
        .collect()
}

// ── Template expansion ────────────────────────────────────────────────────────

/// Expand a template TestCase into N TestCases, one per dataset row.
/// Substitutes `{{field}}` placeholders with row values.
/// Built-in variables: `{{_index}}` (1-based), `{{_total}}`.
pub fn expand_template(
    template: &TestCase,
    rows: &[Row],
    slice_field: Option<&str>,
) -> Vec<TestCase> {
    let total = rows.len();
    rows.iter()
        .enumerate()
        .map(|(i, row)| {
            let mut vars: HashMap<String, String> = row.clone();
            vars.insert("_index".into(), (i + 1).to_string());
            vars.insert("_total".into(), total.to_string());

            // Inherit any vars already on the template (lower priority)
            for (k, v) in &template.vars {
                vars.entry(k.clone()).or_insert_with(|| v.clone());
            }

            let name = apply_vars(&template.name, &vars);
            let prompt = apply_vars(&template.prompt, &vars);
            let context: Vec<String> = template
                .context
                .iter()
                .map(|c| apply_vars(c, &vars))
                .collect();

            // Expand assertion values that contain {{field}} refs
            let assert: Vec<Assertion> = template
                .assert
                .iter()
                .map(|a| expand_assertion(a, &vars))
                .collect();

            // Attach slice field as a var so report can group by it
            let mut final_vars = vars.clone();
            if let Some(sf) = slice_field {
                if let Some(val) = row.get(sf) {
                    final_vars.insert("_slice".into(), val.clone());
                }
            }

            TestCase {
                name,
                description: template.description.clone(),
                prompt,
                context,
                vars: final_vars,
                assert,
                // Inherit other fields from template
                ..template.clone()
            }
        })
        .collect()
}

/// Substitute {{field}} in assertion string values (contains, regex, etc.)
fn expand_assertion(a: &Assertion, vars: &HashMap<String, String>) -> Assertion {
    match a {
        Assertion::Contains { value, weight } => Assertion::Contains {
            value: apply_vars(value, vars),
            weight: *weight,
        },
        Assertion::NotContains { value, weight } => Assertion::NotContains {
            value: apply_vars(value, vars),
            weight: *weight,
        },
        Assertion::Regex { value, weight } => Assertion::Regex {
            value: apply_vars(value, vars),
            weight: *weight,
        },
        Assertion::ExactMatch { value, weight } => Assertion::ExactMatch {
            value: apply_vars(value, vars),
            weight: *weight,
        },
        Assertion::LlmJudge {
            rubric,
            weight,
            threshold,
        } => Assertion::LlmJudge {
            rubric: apply_vars(rubric, vars),
            weight: *weight,
            threshold: *threshold,
        },
        Assertion::Semantic {
            reference,
            threshold,
            weight,
        } => Assertion::Semantic {
            reference: apply_vars(reference, vars),
            threshold: *threshold,
            weight: *weight,
        },
        // All other assertion types pass through unchanged
        other => other.clone(),
    }
}

fn apply_vars(s: &str, vars: &HashMap<String, String>) -> String {
    let mut out = s.to_string();
    for (k, v) in vars {
        out = out.replace(&format!("{{{{{k}}}}}"), v);
    }
    out
}

// ── Aggregate statistics ──────────────────────────────────────────────────────

/// Compute aggregate stats from a completed dataset run.
pub fn compute_stats(
    results: &[crate::runner::TestResult],
    rows: &[Row],
    slice_field: Option<&str>,
) -> DatasetStats {
    let total = results.len();
    let passed = results.iter().filter(|r| r.passed).count();
    let avg_score = if total == 0 {
        0.0
    } else {
        results.iter().map(|r| r.score).sum::<f64>() / total as f64
    };

    // Percentile scores
    let mut scores: Vec<f64> = results.iter().map(|r| r.score).collect();
    scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p10 = percentile(&scores, 0.10);
    let p50 = percentile(&scores, 0.50);
    let p90 = percentile(&scores, 0.90);

    // TTFT percentiles
    let mut ttfts: Vec<u64> = results
        .iter()
        .filter(|r| r.ttft_ms > 0)
        .map(|r| r.ttft_ms)
        .collect();
    ttfts.sort();
    let ttft_p50_ms = u64_percentile(&ttfts, 0.50);
    let ttft_p90_ms = u64_percentile(&ttfts, 0.90);

    // Score histogram (5 buckets: 0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0)
    let mut histogram = [0usize; 5];
    for s in &scores {
        let bucket = ((*s * 5.0).floor() as usize).min(4);
        histogram[bucket] += 1;
    }

    // Worst failures (up to 5)
    let mut indexed: Vec<(usize, &crate::runner::TestResult)> =
        results.iter().enumerate().collect();
    indexed.sort_by(|a, b| a.1.score.partial_cmp(&b.1.score).unwrap());
    let failures: Vec<RowFailure> = indexed
        .iter()
        .filter(|(_, r)| !r.passed)
        .take(5)
        .map(|(i, r)| {
            let input = rows
                .get(*i)
                .and_then(|row| row.get("input"))
                .map(|s| truncate(s, 80))
                .unwrap_or_default();
            let expected = rows
                .get(*i)
                .and_then(|row| row.get("expected"))
                .map(|s| truncate(s, 60))
                .unwrap_or_default();
            RowFailure {
                index: i + 1,
                score: r.score,
                input,
                output: truncate(&r.output, 80),
                expected,
                reason: r.reason.clone(),
            }
        })
        .collect();

    // Slice analysis
    let slices = if let Some(sf) = slice_field {
        compute_slices(results, rows, sf)
    } else {
        vec![]
    };

    DatasetStats {
        total,
        passed,
        avg_score,
        p10,
        p50,
        p90,
        ttft_p50_ms,
        ttft_p90_ms,
        slices,
        failures,
        histogram,
    }
}

fn compute_slices(
    results: &[crate::runner::TestResult],
    rows: &[Row],
    slice_field: &str,
) -> Vec<(String, usize, usize, f64)> {
    let mut groups: HashMap<String, Vec<f64>> = HashMap::new();
    for (i, r) in results.iter().enumerate() {
        let key = rows
            .get(i)
            .and_then(|row| row.get(slice_field))
            .cloned()
            .unwrap_or_else(|| "unknown".into());
        groups.entry(key).or_default().push(r.score);
    }

    let mut slices: Vec<(String, usize, usize, f64)> = groups
        .into_iter()
        .map(|(name, scores)| {
            let total = scores.len();
            let passed = scores.iter().filter(|&&s| s >= 0.5).count();
            let avg = scores.iter().sum::<f64>() / total as f64;
            (name, passed, total, avg)
        })
        .collect();
    // Sort by avg score ascending (weakest slices first)
    slices.sort_by(|a, b| a.3.partial_cmp(&b.3).unwrap());
    slices
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn u64_percentile(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn truncate(s: &str, max: usize) -> String {
    let s = s.replace('\n', " ");
    if s.len() <= max {
        s
    } else {
        format!("{}…", &s[..max])
    }
}

// ── Terminal report ───────────────────────────────────────────────────────────

pub fn print_stats(
    stats: &DatasetStats,
    model: &str,
    dataset_path: &str,
    slice_field: Option<&str>,
) {
    println!();
    println!("{}", "═".repeat(70).bold());
    println!("{:^70}", format!("📊  DATASET RESULTS  —  {model}").bold());
    println!("{}", "═".repeat(70).bold());
    println!("  Dataset   {}", dataset_path.dimmed());
    println!();

    // ── Pass rate ─────────────────────────────────────────────────────────────
    let pass_rate = if stats.total == 0 {
        0.0
    } else {
        stats.passed as f64 / stats.total as f64 * 100.0
    };
    let pass_str = format!("{}/{}", stats.passed, stats.total);
    let rate_str = format!("{pass_rate:.1}%");
    let rate_colored = if pass_rate >= 80.0 {
        rate_str.green()
    } else if pass_rate >= 60.0 {
        rate_str.yellow()
    } else {
        rate_str.red()
    };

    println!("  Pass rate    {} {}", pass_str.bold(), rate_colored);
    println!("  Avg score    {:.3}", stats.avg_score);
    println!(
        "  Percentiles  p10={:.3}  p50={:.3}  p90={:.3}",
        stats.p10, stats.p50, stats.p90
    );
    if stats.ttft_p50_ms > 0 {
        println!(
            "  TTFT         p50={}ms  p90={}ms",
            stats.ttft_p50_ms, stats.ttft_p90_ms
        );
    }

    // ── Score histogram ───────────────────────────────────────────────────────
    println!();
    println!("  {}", "SCORE DISTRIBUTION".bold());
    let max_count = *stats.histogram.iter().max().unwrap_or(&1).max(&1);
    let bar_width = 30usize;
    let labels = ["0.0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"];
    for (i, &count) in stats.histogram.iter().enumerate() {
        let filled = (count * bar_width) / max_count;
        let bar: String = "█".repeat(filled) + &"░".repeat(bar_width - filled);
        let bar_colored = if i >= 3 {
            bar.green()
        } else if i == 2 {
            bar.yellow()
        } else {
            bar.red()
        };
        println!("  {}  {}  {}", labels[i].dimmed(), bar_colored, count);
    }

    // ── Slice analysis ────────────────────────────────────────────────────────
    if !stats.slices.is_empty() {
        println!();
        println!(
            "  {} (by {})",
            "SLICE ANALYSIS".bold(),
            slice_field.unwrap_or("?").cyan()
        );
        println!("  {}", "─".repeat(60).dimmed());
        for (name, passed, total, avg) in &stats.slices {
            let pct = *passed as f64 / *total as f64 * 100.0;
            let pct_s = format!("{pct:.0}%");
            let pct_c = if pct >= 80.0 {
                pct_s.green()
            } else if pct >= 60.0 {
                pct_s.yellow()
            } else {
                pct_s.red()
            };
            let weak = if pct < 60.0 {
                " ← weak".red().to_string()
            } else {
                String::new()
            };
            println!(
                "  {:<24}  {}/{} ({})  avg {:.3}{}",
                name.bold(),
                passed,
                total,
                pct_c,
                avg,
                weak
            );
        }
    }

    // ── Top failures ─────────────────────────────────────────────────────────
    if !stats.failures.is_empty() {
        println!();
        println!("  {}", "TOP FAILURES".bold());
        println!("  {}", "─".repeat(60).dimmed());
        for f in &stats.failures {
            println!("  row-{}  score={:.2}", f.index, f.score);
            if !f.input.is_empty() {
                println!("    input:    {}", f.input.dimmed());
            }
            if !f.expected.is_empty() {
                println!("    expected: {}", f.expected.dimmed());
            }
            if !f.output.is_empty() {
                println!("    got:      {}", f.output.yellow());
            }
            println!("    reason:   {}", f.reason.red());
            println!();
        }
    }

    println!("{}", "═".repeat(70).bold());
    println!();
}
