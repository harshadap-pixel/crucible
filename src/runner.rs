use anyhow::Result;
use chrono::Utc;
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Semaphore;
use uuid::Uuid;

use crate::assertions;
use crate::cli::RunArgs;
use crate::config::{self, HttpConfig, ScriptConfig, TestCase};
use crate::providers::{script as script_provider, OllamaClient};
use crate::report;
use crate::store::{
    db::{RunRecord, TestRecord},
    Db,
};

// ── Public result type ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct TestResult {
    pub test_name: String,
    pub score: f64,
    pub passed: bool,
    pub pass_rate: f64, // fraction of n_runs that passed
    pub latency_ms: u64,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub reason: String,
    pub description: Option<String>,
}

// ── Public entry point ────────────────────────────────────────────────────────

pub async fn run(args: RunArgs) -> Result<()> {
    let cli_vars = parse_vars(&args.vars)?;
    let suite_paths = resolve_suites(&args)?;
    if suite_paths.is_empty() {
        anyhow::bail!("No suite files found. Check --suite / --dir path.");
    }

    let client = Arc::new(OllamaClient::new(&args.ollama_url));

    for path in &suite_paths {
        let stopped = run_suite(path, &args, &cli_vars, &client).await?;
        if args.fail_fast && stopped {
            break;
        }
    }

    Ok(())
}

// ── Suite resolution ──────────────────────────────────────────────────────────

fn resolve_suites(args: &RunArgs) -> Result<Vec<String>> {
    if let Some(ref dir) = args.dir {
        let paths = collect_toml_files(dir);
        if let Some(ref cat) = args.category {
            let mut matched = Vec::new();
            for p in paths {
                if let Ok(s) = config::load(&p) {
                    if s.suite.category.as_deref() == Some(cat.as_str()) {
                        matched.push(p);
                    }
                }
            }
            Ok(matched)
        } else {
            Ok(paths)
        }
    } else {
        Ok(vec![args.suite.clone()])
    }
}

fn collect_toml_files(dir: &str) -> Vec<String> {
    let mut out = Vec::new();
    collect_toml_recursive(std::path::Path::new(dir), &mut out);
    out.sort();
    out
}

fn collect_toml_recursive(dir: &std::path::Path, out: &mut Vec<String>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_toml_recursive(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("toml") {
            out.push(path.display().to_string());
        }
    }
}

// ── Single-suite runner ───────────────────────────────────────────────────────

async fn run_suite(
    suite_path: &str,
    args: &RunArgs,
    cli_vars: &HashMap<String, String>,
    client: &Arc<OllamaClient>,
) -> Result<bool> {
    let mut suite = config::load(suite_path)?;
    if let Some(ref m) = args.model {
        suite.suite.model = m.clone();
    }
    if let Some(ref j) = args.judge {
        suite.suite.judge = j.clone();
    }

    // Directory of the suite file — used to resolve snapshot paths
    let suite_dir = std::path::Path::new(suite_path)
        .parent()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| ".".to_string());

    // --filter: keep only tests whose name contains the pattern
    if let Some(ref pat) = args.filter {
        suite.tests.retain(|t| t.name.contains(pat.as_str()));
        if suite.tests.is_empty() {
            eprintln!(
                "  {} No tests match --filter '{pat}' in {suite_path}",
                "⚠".yellow()
            );
            return Ok(false);
        }
    }

    let ollama_url = args.ollama_url.clone();
    // HTTP provider and script provider skip Ollama health check
    let is_ollama =
        suite.suite.script.is_none() && suite.suite.http.is_none() && suite.suite.model != "mock";

    if is_ollama && !client.health().await {
        anyhow::bail!("Ollama not reachable at {ollama_url}. Is it running?");
    }

    // MoE n_runs multiplier (Ollama only)
    let suite_n_runs = if !is_ollama {
        args.n_runs.max(suite.suite.n_runs)
    } else {
        let info = client.show(&suite.suite.model).await.unwrap_or_default();
        let meta = crate::detect::model_meta::ModelMetadata::from_ollama(&info);
        let mult = meta.n_runs_multiplier();
        let base = args.n_runs.max(suite.suite.n_runs);
        let final_n = base * mult;
        if mult > 1 {
            println!("  {} MoE model — n_runs scaled to {final_n}", "⚠".yellow());
        }
        final_n
    };

    let total = suite.tests.len();
    println!(
        "\n{} {} — {} test(s) — model: {}",
        "RUNNING".bold().cyan(),
        suite.suite.name.bold(),
        total,
        suite.suite.model.yellow(),
    );
    println!("{}", "─".repeat(62).dimmed());

    let bar = ProgressBar::new(total as u64);
    bar.set_style(
        ProgressStyle::with_template("  [{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("█▉▊▋▌▍▎▏  "),
    );

    let http_cfg = suite.suite.http.clone();
    let sem = Arc::new(Semaphore::new(args.concurrency));
    let model = Arc::new(suite.suite.model.clone());
    let judge = Arc::new(suite.suite.judge.clone());
    let script_arc = Arc::new(suite.suite.script.clone());
    let http_arc = Arc::new(http_cfg);
    let cli_vars_arc = Arc::new(cli_vars.clone());
    let suite_dir_arc = Arc::new(suite_dir.clone());
    let update_snapshots = args.update_snapshots;
    let retry = args.retry;
    let fail_fast = args.fail_fast;
    let mut handles = Vec::new();

    for test in suite.tests.clone() {
        let client = client.clone();
        let model = model.clone();
        let judge = judge.clone();
        let script_arc = script_arc.clone();
        let http_arc = http_arc.clone();
        let sem = sem.clone();
        let bar = bar.clone();
        let extra_vars = cli_vars_arc.clone();
        let suite_dir = suite_dir_arc.clone();

        // Per-test n_runs takes priority over suite-level
        let effective_n = test.n_runs.unwrap_or(suite_n_runs);

        handles.push(tokio::spawn(async move {
            let _permit = sem.acquire().await.unwrap();
            let result = run_test_with_retry(
                &client,
                &model,
                &judge,
                &test,
                effective_n,
                &script_arc,
                &http_arc,
                &extra_vars,
                &suite_dir,
                update_snapshots,
                retry,
            )
            .await;
            bar.inc(1);
            bar.set_message(test.name.clone());
            result
        }));
    }

    let mut results: Vec<TestResult> = Vec::new();
    let mut stopped = false;

    for h in handles {
        match h.await? {
            Ok(r) => {
                let failed = !r.passed;
                results.push(r);
                if fail_fast && failed {
                    stopped = true;
                    break;
                }
            }
            Err(e) => eprintln!("  {} Test error: {e}", "✗".red()),
        }
    }
    bar.finish_and_clear();

    // ── Persist results ───────────────────────────────────────────────────────
    let run_id = Uuid::new_v4().to_string()[..8].to_string();
    let timestamp = Utc::now().format("%Y-%m-%d %H:%M:%S").to_string();
    let passed = results.iter().filter(|r| r.passed).count() as u32;
    let avg_score = if results.is_empty() {
        0.0
    } else {
        results.iter().map(|r| r.score).sum::<f64>() / results.len() as f64
    };
    let total_tokens: u64 = results
        .iter()
        .map(|r| r.input_tokens as u64 + r.output_tokens as u64)
        .sum();

    let db = Db::open()?;
    db.insert_run(&RunRecord {
        id: run_id.clone(),
        suite_name: suite.suite.name.clone(),
        model: suite.suite.model.clone(),
        timestamp: timestamp.clone(),
        is_baseline: false,
        total_tests: total as u32,
        passed_tests: passed,
        avg_score,
        total_tokens,
    })?;

    for r in &results {
        db.insert_test_result(&TestRecord {
            run_id: run_id.clone(),
            test_name: r.test_name.clone(),
            score: r.score,
            passed: r.passed,
            pass_rate: r.pass_rate,
            latency_ms: r.latency_ms,
            input_tokens: r.input_tokens,
            output_tokens: r.output_tokens,
            reason: r.reason.clone(),
        })?;
    }

    // ── Per-suite baseline auto-set ───────────────────────────────────────────
    if args.baseline {
        db.set_baseline(&run_id, &suite.suite.name)?;
        println!(
            "  {} Run {run_id} marked as baseline for '{}'.",
            "●".yellow(),
            suite.suite.name
        );
    }

    // ── Per-suite baseline comparison ─────────────────────────────────────────
    let baseline_results = if args.compare {
        db.current_baseline(&suite.suite.name)?
            .map(|b| db.get_results_for_run(&b.id))
            .transpose()?
    } else {
        None
    };

    // ── Output ────────────────────────────────────────────────────────────────
    match args.output.as_str() {
        "json" => print_json(
            &run_id,
            &timestamp,
            &suite.suite.model,
            &suite.suite.name,
            &results,
        ),
        "sarif" => print_sarif(&suite.suite.name, &results),
        _ => {
            report::print_run(
                &run_id,
                &timestamp,
                &suite.suite.model,
                &suite.suite.name,
                &results,
                baseline_results.as_deref(),
                suite.suite.regression_threshold,
            );
            if total_tokens > 0 {
                println!(
                    "  Tokens: {} in / {} out (total {})",
                    results.iter().map(|r| r.input_tokens as u64).sum::<u64>(),
                    results.iter().map(|r| r.output_tokens as u64).sum::<u64>(),
                    total_tokens,
                );
            }
            println!("  Run ID: {}", run_id.dimmed());
            if !args.baseline {
                println!("  Tip: re-run with --baseline to pin this as the regression baseline\n");
            }
        }
    }

    Ok(stopped)
}

// ── Output: JSON ──────────────────────────────────────────────────────────────

fn print_json(
    run_id: &str,
    timestamp: &str,
    model: &str,
    suite_name: &str,
    results: &[TestResult],
) {
    let passed = results.iter().filter(|r| r.passed).count();
    let avg = if results.is_empty() {
        0.0
    } else {
        results.iter().map(|r| r.score).sum::<f64>() / results.len() as f64
    };
    let total_tokens: u64 = results
        .iter()
        .map(|r| r.input_tokens as u64 + r.output_tokens as u64)
        .sum();

    let tests_json: Vec<String> = results
        .iter()
        .map(|r| {
            format!(
                "    {{\"name\":{},\"passed\":{},\"score\":{:.4},\"pass_rate\":{:.4},\
             \"latency_ms\":{},\"input_tokens\":{},\"output_tokens\":{},\"reason\":{}}}",
                json_str(&r.test_name),
                r.passed,
                r.score,
                r.pass_rate,
                r.latency_ms,
                r.input_tokens,
                r.output_tokens,
                json_str(&r.reason),
            )
        })
        .collect();

    println!(
        "{{\"run_id\":{},\"timestamp\":{},\"model\":{},\"suite\":{},\
         \"passed\":{passed},\"total\":{},\"avg_score\":{avg:.4},\
         \"total_tokens\":{total_tokens},\"tests\":[\n{}\n]}}",
        json_str(run_id),
        json_str(timestamp),
        json_str(model),
        json_str(suite_name),
        results.len(),
        tests_json.join(",\n"),
    );
}

// ── Output: SARIF ─────────────────────────────────────────────────────────────

fn print_sarif(suite_name: &str, results: &[TestResult]) {
    let _ = suite_name;

    let rules: String = results
        .iter()
        .map(|r| {
            let id = json_str(&r.test_name);
            let desc = json_str(r.description.as_deref().unwrap_or(&r.test_name));
            let mut s = String::from("      {");
            s.push_str(&format!(
                "\"id\":{id},\"name\":{id},\"shortDescription\":{{\"text\":{desc}}}"
            ));
            s.push('}');
            s
        })
        .collect::<Vec<_>>()
        .join(",\n");

    let sarif_results: String = results
        .iter()
        .map(|r| {
            let level = if r.passed { "none" } else { "warning" };
            let rule = json_str(&r.test_name);
            let lvl = json_str(level);
            let reason = json_str(&r.reason);
            let mut s = String::from("      {");
            s.push_str(&format!(
                "\"ruleId\":{rule},\"level\":{lvl},\"message\":{{\"text\":{reason}}}"
            ));
            s.push('}');
            s
        })
        .collect::<Vec<_>>()
        .join(",\n");

    let mut out = String::new();
    out.push_str("{\n");
    out.push_str("  \"$schema\":\"https://json.schemastore.org/sarif-2.1.0.json\",\n");
    out.push_str("  \"version\":\"2.1.0\",\n");
    out.push_str("  \"runs\":[{\n");
    out.push_str(
        "    \"tool\":{\"driver\":{\"name\":\"crucible\",\"version\":\"0.1.0\",\"rules\":[\n",
    );
    out.push_str(&rules);
    out.push_str("\n    ]}},\n");
    out.push_str("    \"results\":[\n");
    out.push_str(&sarif_results);
    out.push_str("\n    ]\n  }]\n}");
    println!("{out}");
}

fn json_str(s: &str) -> String {
    format!(
        "\"{}\"",
        s.replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
    )
}

// ── --var parsing ─────────────────────────────────────────────────────────────

fn parse_vars(raw: &[String]) -> Result<HashMap<String, String>> {
    let mut map = HashMap::new();
    for kv in raw {
        let (k, v) = kv
            .split_once('=')
            .ok_or_else(|| anyhow::anyhow!("--var must be KEY=VALUE, got: {kv}"))?;
        map.insert(k.to_string(), v.to_string());
    }
    Ok(map)
}

// ── Test execution ────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
async fn run_test_with_retry(
    client: &OllamaClient,
    model: &str,
    judge: &str,
    test: &TestCase,
    n_runs: u32,
    script_cfg: &Option<ScriptConfig>,
    http_cfg: &Option<HttpConfig>,
    cli_vars: &HashMap<String, String>,
    suite_dir: &str,
    update_snapshots: bool,
    retry: u32,
) -> Result<TestResult> {
    let mut last_err = None;
    for attempt in 0..=retry {
        match run_test(
            client,
            model,
            judge,
            test,
            n_runs,
            script_cfg,
            http_cfg,
            cli_vars,
            suite_dir,
            update_snapshots,
        )
        .await
        {
            Ok(r) => return Ok(r),
            Err(e) => {
                last_err = Some(e);
                if attempt < retry {
                    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                }
            }
        }
    }
    Err(last_err.unwrap())
}

/// Provider priority:
///   1. HTTP  (http_cfg present + test has method + path)
///   2. Script (script_cfg present)
///   3. Multi-turn Ollama (test.turns non-empty)
///   4. Single-turn Ollama (default)
#[allow(clippy::too_many_arguments)]
async fn run_test(
    client: &OllamaClient,
    model: &str,
    judge: &str,
    test: &TestCase,
    n_runs: u32,
    script_cfg: &Option<ScriptConfig>,
    http_cfg: &Option<HttpConfig>,
    cli_vars: &HashMap<String, String>,
    suite_dir: &str,
    update_snapshots: bool,
) -> Result<TestResult> {
    // Merge suite vars + CLI vars (CLI wins on conflict)
    let mut vars = test.vars.clone();
    vars.extend(cli_vars.iter().map(|(k, v)| (k.clone(), v.clone())));
    let prompt = apply_vars(&test.prompt, &vars);

    let full_prompt = if test.context.is_empty() {
        prompt.clone()
    } else {
        format!(
            "Context:\n{}\n\nQuestion: {prompt}",
            test.context.join("\n---\n")
        )
    };

    let mut scores = Vec::new();
    let mut passed_n = 0u32;
    let mut last_reason = String::new();
    let mut total_latency = 0u64;
    let mut total_in_tok = 0u32;
    let mut total_out_tok = 0u32;

    for _ in 0..n_runs.max(1) {
        let (output, latency_ms, http_status, in_tok, out_tok) =
            if let (Some(hc), Some(method), Some(path)) = (http_cfg, &test.method, &test.path) {
                // ── HTTP provider ──────────────────────────────────────────────
                let url = format!("{}{}", hc.base_url.trim_end_matches('/'), path);
                let mut req = match method.to_uppercase().as_str() {
                    "POST" => reqwest::Client::new().post(&url),
                    "PUT" => reqwest::Client::new().put(&url),
                    "DELETE" => reqwest::Client::new().delete(&url),
                    _ => reqwest::Client::new().get(&url),
                };
                for (k, v) in &hc.headers {
                    req = req.header(k, v);
                }
                for (k, v) in &test.headers {
                    req = req.header(k, v);
                }
                if let Some(body) = &test.body {
                    req = req
                        .header("Content-Type", "application/json")
                        .body(body.clone());
                }

                let t0 = std::time::Instant::now();
                let resp = req
                    .timeout(std::time::Duration::from_millis(hc.timeout_ms))
                    .send()
                    .await?;
                let lat = t0.elapsed().as_millis() as u64;
                let code = resp.status().as_u16();

                // ── SSE accumulation ───────────────────────────────────────────
                // Triggered by: explicit test.sse = true  OR  Content-Type: text/event-stream
                let is_sse = test.sse
                    || resp
                        .headers()
                        .get("content-type")
                        .and_then(|v| v.to_str().ok())
                        .map(|v| v.contains("text/event-stream"))
                        .unwrap_or(false);

                let body = if is_sse {
                    let timeout_ms = test.sse_timeout_ms.unwrap_or(hc.timeout_ms);
                    accumulate_sse(resp, timeout_ms).await
                } else {
                    resp.text().await.unwrap_or_default()
                };

                (body, lat, Some(code), 0u32, 0u32)
            } else if let Some(sc) = script_cfg {
                // ── Script provider ────────────────────────────────────────────
                let args2 = sc.args.clone();
                let env2: Vec<(String, String)> = sc
                    .env
                    .iter()
                    .map(|p| (p[0].clone(), p[1].clone()))
                    .collect();
                let cmd = sc.cmd.clone();
                let p = full_prompt.clone();
                let sr = tokio::task::spawn_blocking(move || {
                    let args_ref: Vec<&str> = args2.iter().map(|s| s.as_str()).collect();
                    let env_ref: Vec<(&str, &str)> =
                        env2.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
                    script_provider::run(&cmd, &args_ref, &p, &env_ref)
                })
                .await??;
                (sr.stdout, sr.latency_ms as u64, None, 0u32, 0u32)
            } else if !test.turns.is_empty() {
                // ── Multi-turn Ollama ──────────────────────────────────────────
                let history: Vec<(String, String)> = test
                    .turns
                    .iter()
                    .map(|t| (t.role.clone(), t.content.clone()))
                    .collect();
                let comp = client
                    .chat_with_history(model, &history, &full_prompt, 0.0)
                    .await?;
                (
                    comp.text,
                    comp.latency_ms as u64,
                    None,
                    comp.input_tokens,
                    comp.output_tokens,
                )
            } else {
                // ── Single-turn Ollama (default) ───────────────────────────────
                let comp = client.chat(model, None, &full_prompt, 0.0).await?;
                (
                    comp.text,
                    comp.latency_ms as u64,
                    None,
                    comp.input_tokens,
                    comp.output_tokens,
                )
            };

        let (score, assertion_results) = assertions::evaluate_all(
            test,
            &output,
            latency_ms,
            http_status,
            client,
            judge,
            suite_dir,
            update_snapshots,
        )
        .await?;

        let passed = assertion_results.iter().all(|a| a.passed);
        let reason = assertion_results
            .iter()
            .filter(|a| !a.passed)
            .map(|a| format!("[{}] {}", a.kind, a.reason))
            .collect::<Vec<_>>()
            .join(" | ");
        let reason = if reason.is_empty() {
            "All assertions passed".into()
        } else {
            reason
        };

        scores.push(score);
        if passed {
            passed_n += 1;
        }
        last_reason = reason;
        total_latency += latency_ms;
        total_in_tok += in_tok;
        total_out_tok += out_tok;
    }

    let n = n_runs.max(1);
    let avg_score = scores.iter().sum::<f64>() / scores.len() as f64;
    let pass_rate = passed_n as f64 / n as f64;

    Ok(TestResult {
        test_name: test.name.clone(),
        score: avg_score,
        passed: passed_n > n / 2, // majority vote
        pass_rate,
        latency_ms: total_latency / n as u64,
        input_tokens: total_in_tok / n,
        output_tokens: total_out_tok / n,
        reason: last_reason,
        description: test.description.clone(),
    })
}

// ── SSE accumulator ───────────────────────────────────────────────────────────

/// Read a Server-Sent Events response, collecting all `data:` payloads.
/// Returns them joined by newlines. Stops when the stream ends or the
/// timeout (ms) elapses — whichever comes first.
///
/// SSE format (per spec):
///   data: <payload>\n\n
///   data: <continuation>\n
///   event: <type>\n
///   id: <id>\n
///   : <comment>\n
///
/// Only `data:` lines are included in the returned string.
async fn accumulate_sse(resp: reqwest::Response, timeout_ms: u64) -> String {
    use tokio::time::{timeout, Duration};

    let mut lines: Vec<String> = Vec::new();

    let collect = async {
        // reqwest doesn't expose a direct line reader; use bytes() chunks.
        let mut buf = String::new();
        let body = resp;

        // We consume via text() which reads the full body — but for SSE over
        // HTTP/1.1 with a server that closes the connection after events,
        // this works correctly. For long-lived SSE streams, the timeout
        // below will cut it off.
        //
        // If the server keeps the connection open indefinitely, clients
        // should set sse_timeout_ms to a small value (e.g. 3000).
        if let Ok(text) = body.text().await {
            buf = text;
        }

        for line in buf.lines() {
            if let Some(payload) = line.strip_prefix("data:") {
                let payload = payload.trim();
                if !payload.is_empty() && payload != "[DONE]" {
                    lines.push(payload.to_string());
                }
            }
            // Skip: event:, id:, retry:, comments (:), blank lines
        }
    };

    let _ = timeout(Duration::from_millis(timeout_ms), collect).await;
    lines.join("\n")
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn apply_vars(prompt: &str, vars: &HashMap<String, String>) -> String {
    let mut out = prompt.to_string();
    for (k, v) in vars {
        out = out.replace(&format!("{{{{{k}}}}}"), v);
    }
    out
}
