/// Generator — turns scanner findings into runnable test suites and shim scripts.
///
/// For eval runners:  Strategy A — subprocess wrapper, assert on summary line.
/// For NL2SQL funcs:  Strategy C — generates a shim + standard security test cases.
/// For RAG pipelines: generates faithfulness + no-hallucination test cases.
///
/// All suites are written to a temp directory and returned as file paths.
/// No changes to the target codebase.
use anyhow::Result;
use std::fs;
use std::path::{Path, PathBuf};

use crate::discover::scanner::{
    Finding, FindingKind, Language, McpFramework, McpTransport, RagProfile,
};

// ── Public entry point ────────────────────────────────────────────────────────

/// Generate TOML suite files for all findings.
/// Returns list of (suite_path, description) pairs.
///
/// MCP findings are consolidated: one suite per server (root file), with tool
/// counts aggregated from co-located tool-definition files. If no root is
/// detected, all non-root MCP findings collapse into a single representative suite.
pub fn generate(findings: &[Finding], save_dir: Option<&str>) -> Result<Vec<(String, String)>> {
    let out_dir: PathBuf = if let Some(d) = save_dir {
        PathBuf::from(d)
    } else {
        std::env::temp_dir().join("crucible-discover")
    };
    fs::create_dir_all(&out_dir)?;

    let mut generated = Vec::new();
    let mut suite_idx: usize = 0;

    // ── Phase 1: Consolidated MCP generation ─────────────────────────────────
    // Separate MCP findings from the rest to avoid one-suite-per-tool-file bloat.
    let mcp: Vec<&Finding> = findings
        .iter()
        .filter(|f| matches!(&f.kind, FindingKind::McpServer { .. }))
        .collect();

    if !mcp.is_empty() {
        let roots: Vec<&Finding> = mcp
            .iter()
            .filter(|f| matches!(&f.kind, FindingKind::McpServer { is_root: true, .. }))
            .copied()
            .collect();

        let non_roots: Vec<&Finding> = mcp
            .iter()
            .filter(|f| matches!(&f.kind, FindingKind::McpServer { is_root: false, .. }))
            .copied()
            .collect();

        if roots.is_empty() {
            // No root detected — collapse all non-root findings into one representative suite.
            // Use the file with the most tool definitions as the representative.
            let total_tools: usize = non_roots
                .iter()
                .filter_map(|f| match &f.kind {
                    FindingKind::McpServer { tool_count, .. } => Some(*tool_count),
                    _ => None,
                })
                .sum();

            let rep = non_roots
                .iter()
                .max_by_key(|f| match &f.kind {
                    FindingKind::McpServer { tool_count, .. } => *tool_count,
                    _ => 0,
                })
                .copied();

            if let Some(rep) = rep {
                if let FindingKind::McpServer {
                    framework,
                    transports,
                    server_name,
                    ..
                } = &rep.kind
                {
                    match generate_mcp_suite(
                        rep,
                        framework,
                        transports,
                        total_tools,
                        server_name.as_deref(),
                        &out_dir,
                        suite_idx,
                    ) {
                        Ok((path, desc)) => {
                            generated.push((path, desc));
                            suite_idx += 1;
                        }
                        Err(e) => eprintln!("  ⚠ Could not generate MCP suite: {e}"),
                    }
                }
            }
        } else {
            // One suite per root. Aggregate co-located non-root tool counts into each root.
            for root in &roots {
                let root_dir = std::path::Path::new(&root.path)
                    .parent()
                    .map(|p| p.display().to_string())
                    .unwrap_or_default();

                let extra_tools: usize = non_roots
                    .iter()
                    .filter(|f| f.path.starts_with(&root_dir))
                    .filter_map(|f| match &f.kind {
                        FindingKind::McpServer { tool_count, .. } => Some(*tool_count),
                        _ => None,
                    })
                    .sum();

                let own_tools = match &root.kind {
                    FindingKind::McpServer { tool_count, .. } => *tool_count,
                    _ => 0,
                };

                if let FindingKind::McpServer {
                    framework,
                    transports,
                    server_name,
                    ..
                } = &root.kind
                {
                    match generate_mcp_suite(
                        root,
                        framework,
                        transports,
                        own_tools + extra_tools,
                        server_name.as_deref(),
                        &out_dir,
                        suite_idx,
                    ) {
                        Ok((path, desc)) => {
                            generated.push((path, desc));
                            suite_idx += 1;
                        }
                        Err(e) => {
                            eprintln!("  ⚠ Could not generate MCP suite for {}: {e}", root.path)
                        }
                    }
                }
            }
        }
    }

    // ── Phase 2: All non-MCP findings ────────────────────────────────────────
    for finding in findings.iter() {
        if matches!(&finding.kind, FindingKind::McpServer { .. }) {
            continue; // already handled above
        }

        let result = match &finding.kind {
            FindingKind::EvalRunner {
                run_cmd,
                test_count,
                summary_pat,
                ..
            } => generate_eval_runner_suite(
                finding,
                run_cmd,
                *test_count,
                summary_pat,
                &out_dir,
                suite_idx,
            ),
            FindingKind::NL2Sql {
                language,
                function_name,
                file_path,
                tsconfig,
                allowed_schemas,
            } => generate_nl2sql_suite(
                finding,
                language,
                function_name,
                file_path,
                tsconfig.as_deref(),
                allowed_schemas,
                &out_dir,
                suite_idx,
            ),
            FindingKind::RagPipeline { profile } => {
                generate_rag_suite(finding, profile, &out_dir, suite_idx)
            }
            FindingKind::AiService { provider } => {
                generate_ai_service_suite(finding, provider, &out_dir, suite_idx)
            }
            FindingKind::McpServer { .. } => unreachable!("filtered above"),
        };

        match result {
            Ok((path, desc)) => {
                generated.push((path, desc));
                suite_idx += 1;
            }
            Err(e) => eprintln!("  ⚠ Could not generate suite for {}: {e}", finding.path),
        }
    }

    Ok(generated)
}

// ── Strategy A: Eval Runner ───────────────────────────────────────────────────

fn generate_eval_runner_suite(
    finding: &Finding,
    run_cmd: &[String],
    test_count: Option<usize>,
    _summary_pat: &str,
    out_dir: &Path,
    idx: usize,
) -> Result<(String, String)> {
    let basename = PathBuf::from(&finding.path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("eval")
        .to_string();

    let suite_name = format!("Auto — {basename} (eval runner)");
    let count_desc = test_count
        .map(|n| format!("~{n} test cases"))
        .unwrap_or_else(|| "unknown test count".into());

    // Build the TOML args array
    let args_toml = run_cmd[1..]
        .iter()
        .map(|a| format!("  \"{}\",", a.replace('"', "\\\"")))
        .collect::<Vec<_>>()
        .join("\n");

    // Build the summary assertion — use what the runner actually prints
    let expected_pass = test_count
        .map(|n| format!("{n}/{n} passed"))
        .unwrap_or_else(|| "passed".into());

    let content = format!(
        r#"# AUTO-GENERATED by `crucible autodiscover`
# Source: {source}
# Pattern: eval runner ({count_desc})
# Strategy: subprocess wrapper — asserts on stdout summary line
# No modifications to the target codebase required.

[suite]
name                 = "{suite_name}"
model                = "mock"
category             = "autodiscover"
concurrency          = 1
n_runs               = 1
regression_threshold = 0.05

[suite.script]
cmd  = "{cmd}"
args = [
{args_toml}
]

[[tests]]
name        = "full-eval-suite"
description = "Run the complete eval suite; assert on summary line"
prompt      = ""

assert = [
  {{ type = "contains",     value = "{expected_pass}", weight = 5.0 }},
  {{ type = "not_contains", value = "✗",               weight = 3.0 }},
  {{ type = "contains",     value = "Results:",         weight = 1.0 }},
]
"#,
        source = finding.path,
        cmd = run_cmd[0],
    );

    let suite_path = out_dir.join(format!("discover_{idx:02}_{basename}.toml"));
    fs::write(&suite_path, content)?;
    Ok((suite_path.display().to_string(), suite_name))
}

// ── Strategy C: NL2SQL Function ───────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn generate_nl2sql_suite(
    finding: &Finding,
    language: &Language,
    function_name: &str,
    file_path: &str,
    tsconfig: Option<&str>,
    allowed_schemas: &[String],
    out_dir: &Path,
    idx: usize,
) -> Result<(String, String)> {
    // Step 1: generate the shim script
    let shim_path = out_dir.join(format!("discover_{idx:02}_nl2sql_shim.ts"));
    let shim_content = nl2sql_shim_content(function_name, file_path);
    fs::write(&shim_path, shim_content)?;

    // Step 2: build run command for the shim
    let (cmd, args_toml) = match language {
        Language::TypeScript => {
            let mut args = vec!["ts-node".to_string()];
            if let Some(tc) = tsconfig {
                args.push("-P".into());
                args.push(tc.into());
            }
            args.push("--transpile-only".into());
            args.push(shim_path.display().to_string());
            let toml_args = args
                .iter()
                .map(|a| format!("  \"{}\",", a.replace('"', "\\\"")))
                .collect::<Vec<_>>()
                .join("\n");
            ("npx".to_string(), toml_args)
        }
        Language::Python => {
            let py_shim = out_dir.join(format!("discover_{idx:02}_nl2sql_shim.py"));
            fs::write(
                &py_shim,
                python_nl2sql_shim_content(function_name, file_path),
            )?;
            let toml_args = format!("  \"{}\",", py_shim.display());
            ("python3".to_string(), toml_args)
        }
        Language::Unknown => (
            "node".to_string(),
            format!("  \"{}\",", shim_path.display()),
        ),
    };

    // Step 3: pick valid + reject SQL based on detected schemas
    let primary_schema = allowed_schemas
        .iter()
        .find(|s| !s.is_empty())
        .cloned()
        .unwrap_or_else(|| "public".into());

    let suite_name = format!("Auto — {function_name} (NL2SQL security)");

    // Helper: wrap prompt JSON in the correct TOML quoting.
    // If the value contains a single quote (SQL string literals like 'value'),
    // we must use triple-quoted TOML literal strings ('''...''').
    // Otherwise a plain single-quoted literal is fine.
    let q = |s: &str| -> String {
        if s.contains('\'') {
            format!("'''{s}'''")
        } else {
            format!("'{s}'")
        }
    };

    let p_valid1 = format!(
        r#"{{"sql":"SELECT id, created_at FROM {primary_schema}.events WHERE org_id = :orgId LIMIT 100","vendorId":"test_vendor","params":[{{"name":"orgId","value":"test_vendor"}}]}}"#
    );
    let p_valid2 = format!(
        r#"{{"sql":"SELECT event_name, COUNT(*) AS cnt FROM {primary_schema}.events WHERE org_id = :orgId GROUP BY event_name","vendorId":"test_vendor","params":[{{"name":"orgId","value":"test_vendor"}}]}}"#
    );
    let p_info = r#"{"sql":"SELECT table_name FROM information_schema.tables","vendorId":"test_vendor","expectReject":true}"#;
    let p_upd = format!(
        r#"{{"sql":"UPDATE {primary_schema}.events SET status = 1 WHERE org_id = 'test_vendor'","vendorId":"test_vendor","expectReject":true}}"#
    );
    let p_drop = format!(
        r#"{{"sql":"DROP TABLE {primary_schema}.events","vendorId":"test_vendor","expectReject":true}}"#
    );
    let p_pg = r#"{"sql":"SELECT * FROM pg_catalog.pg_tables","vendorId":"test_vendor","expectReject":true}"#;
    let p_sys =
        r#"{"sql":"SELECT * FROM sys.credentials","vendorId":"test_vendor","expectReject":true}"#;
    let p_multi = format!(
        r#"{{"sql":"SELECT 1; DROP TABLE {primary_schema}.events","vendorId":"test_vendor","expectReject":true}}"#
    );

    let content = format!(
        r#"# AUTO-GENERATED by `crucible autodiscover`
# Source: {source}
# Function: {function_name}
# Strategy: direct function call via generated shim (no server, no LLM)
#
# Shim: {shim_path}
# Protocol: stdin JSON → stdout JSON
#   in:  {{ "sql":"...", "vendorId":"...", "expectReject":bool }}
#   out: {{ "pass":true }} | {{ "pass":false, "reason":"..." }}

[suite]
name                 = "{suite_name}"
model                = "mock"
category             = "autodiscover"
concurrency          = 4
n_runs               = 1
regression_threshold = 0.05

[suite.script]
cmd  = "{cmd}"
args = [
{args_toml}
]

# ── Valid queries — must be accepted ──────────────────────────────────────────

[[tests]]
name        = "valid-select-allowed-schema"
description = "Basic SELECT on allowed schema with parameterised org_id — must pass"
prompt      = {pv1}

assert = [
  {{ type = "contains",     value = '"pass":true',  weight = 5.0 }},
  {{ type = "not_contains", value = '"pass":false'               }},
]

[[tests]]
name        = "valid-select-with-group-by"
description = "Aggregation query on allowed schema — must pass"
prompt      = {pv2}

assert = [
  {{ type = "contains",     value = '"pass":true',  weight = 5.0 }},
  {{ type = "not_contains", value = '"pass":false'               }},
]

# ── Rejection cases — must be BLOCKED ────────────────────────────────────────
# A pass:false here means the security layer let a dangerous query through.

[[tests]]
name        = "rejects-information-schema"
description = "Schema exfiltration via information_schema — must throw"
prompt      = {pi}

assert = [
  {{ type = "contains",     value = '"pass":true',  weight = 5.0 }},
  {{ type = "not_contains", value = '"pass":false',  weight = 5.0 }},
  {{ type = "contains",     value = '"reason":',     weight = 1.0 }},
]

[[tests]]
name        = "rejects-update-mutation"
description = "DML mutation (UPDATE) — must throw"
prompt      = {pu}

assert = [
  {{ type = "contains",     value = '"pass":true',  weight = 5.0 }},
  {{ type = "not_contains", value = '"pass":false',  weight = 5.0 }},
  {{ type = "contains",     value = '"reason":',     weight = 1.0 }},
]

[[tests]]
name        = "rejects-drop-table"
description = "DDL DROP TABLE — must throw"
prompt      = {pd}

assert = [
  {{ type = "contains",     value = '"pass":true',  weight = 5.0 }},
  {{ type = "not_contains", value = '"pass":false',  weight = 5.0 }},
  {{ type = "contains",     value = '"reason":',     weight = 1.0 }},
]

[[tests]]
name        = "rejects-pg-catalog"
description = "System table access via pg_catalog — must throw"
prompt      = {ppg}

assert = [
  {{ type = "contains",     value = '"pass":true',  weight = 5.0 }},
  {{ type = "not_contains", value = '"pass":false',  weight = 5.0 }},
  {{ type = "contains",     value = '"reason":',     weight = 1.0 }},
]

[[tests]]
name        = "rejects-sys-schema"
description = "System schema access (sys.*) — must throw"
prompt      = {ps}

assert = [
  {{ type = "contains",     value = '"pass":true',  weight = 5.0 }},
  {{ type = "not_contains", value = '"pass":false',  weight = 5.0 }},
  {{ type = "contains",     value = '"reason":',     weight = 1.0 }},
]

[[tests]]
name        = "rejects-multi-statement"
description = "Multi-statement injection via semicolon — must throw"
prompt      = {pm}

assert = [
  {{ type = "contains",     value = '"pass":true',  weight = 5.0 }},
  {{ type = "not_contains", value = '"pass":false',  weight = 5.0 }},
  {{ type = "contains",     value = '"reason":',     weight = 1.0 }},
]
"#,
        source = finding.path,
        shim_path = shim_path.display(),
        pv1 = q(&p_valid1),
        pv2 = q(&p_valid2),
        pi = q(p_info),
        pu = q(&p_upd),
        pd = q(&p_drop),
        ppg = q(p_pg),
        ps = q(p_sys),
        pm = q(&p_multi),
    );

    let suite_path = out_dir.join(format!("discover_{idx:02}_nl2sql_security.toml"));
    fs::write(&suite_path, content)?;
    Ok((suite_path.display().to_string(), suite_name))
}

// ── RAG suite — strategy-aware ────────────────────────────────────────────────

fn generate_rag_suite(
    finding: &Finding,
    profile: &RagProfile,
    out_dir: &Path,
    idx: usize,
) -> Result<(String, String)> {
    let basename = PathBuf::from(&finding.path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("rag")
        .to_string();

    let strategy_summary = profile.summary();
    let suite_name = format!("Auto — {basename} (RAG — {strategy_summary})");

    // ── Header comment lines per detected strategy ────────────────────────────
    let mut strategy_lines = String::new();
    if !profile.frameworks.is_empty() {
        strategy_lines.push_str(&format!(
            "# Framework:  {}\n",
            profile.frameworks.join(", ")
        ));
    }
    if !profile.retrieval.is_empty() {
        strategy_lines.push_str(&format!("# Retrieval:  {}\n", profile.retrieval.join(", ")));
    }
    if !profile.chunking.is_empty() {
        strategy_lines.push_str(&format!("# Chunking:   {}\n", profile.chunking.join(", ")));
    }
    if !profile.reranking.is_empty() {
        strategy_lines.push_str(&format!("# Reranking:  {}\n", profile.reranking.join(", ")));
    }
    if !profile.embedding.is_empty() {
        strategy_lines.push_str(&format!("# Embedding:  {}\n", profile.embedding.join(", ")));
    }

    let n_strategy_tests = (!profile.chunking.is_empty() as usize)
        + (!profile.reranking.is_empty() as usize)
        + (profile.retrieval.iter().any(|s| s == "hybrid") as usize) * 2
        + (profile.retrieval.iter().any(|s| s == "hyde") as usize)
        + (profile.retrieval.iter().any(|s| s == "multi_query") as usize)
        + (profile.chunking.iter().any(|s| s == "late_chunking") as usize)
        + (profile.retrieval.iter().any(|s| s == "parent_doc") as usize)
        + (profile
            .retrieval
            .iter()
            .any(|s| s == "contextual_compression") as usize);

    let total_tests = 2 + n_strategy_tests;

    let header = format!(
        "# AUTO-GENERATED by `crucible autodiscover`\n\
         # Source: {source}\n\
         # Pattern: RAG pipeline ({total_tests} tests — 2 base + {n_strategy_tests} strategy-specific)\n\
         {strategy_lines}\
         # Update model/judge and adapt prompts/contexts to your actual data schema.\n\
         \n\
         [suite]\n\
         name                 = \"{suite_name}\"\n\
         model                = \"\"\n\
         judge                = \"\"\n\
         category             = \"autodiscover\"\n\
         concurrency          = 2\n\
         n_runs               = 1\n\
         regression_threshold = 0.10\n",
        source = finding.path,
    );

    // ── Build test body ───────────────────────────────────────────────────────
    let mut body = String::from(RAG_BASE_TESTS);

    if !profile.chunking.is_empty() {
        body.push_str(rag_chunk_boundary_test());
    }
    if !profile.reranking.is_empty() {
        body.push_str(rag_reranker_sensitivity_test());
    }
    if profile.retrieval.iter().any(|s| s == "hybrid") {
        body.push_str(rag_hybrid_keyword_test());
        body.push_str(rag_hybrid_semantic_test());
    }
    if profile.retrieval.iter().any(|s| s == "hyde") {
        body.push_str(rag_hyde_test());
    }
    if profile.retrieval.iter().any(|s| s == "multi_query") {
        body.push_str(rag_multi_query_test());
    }
    if profile.chunking.iter().any(|s| s == "late_chunking") {
        body.push_str(rag_late_chunking_test());
    }
    if profile.retrieval.iter().any(|s| s == "parent_doc") {
        body.push_str(rag_parent_doc_test());
    }
    if profile
        .retrieval
        .iter()
        .any(|s| s == "contextual_compression")
    {
        body.push_str(rag_contextual_compression_test());
    }

    let content = format!("{header}{body}");
    let suite_path = out_dir.join(format!("discover_{idx:02}_{basename}_rag.toml"));
    fs::write(&suite_path, content)?;
    Ok((suite_path.display().to_string(), suite_name))
}

// ── Base tests (always emitted) ───────────────────────────────────────────────

const RAG_BASE_TESTS: &str = "\n\
[[tests]]\n\
name        = \"rag-faithfulness\"\n\
description = \"Response must be grounded in the provided context\"\n\
context     = [\"The system processed 42 events on 2026-01-15.\"]\n\
prompt      = \"How many events were processed on 2026-01-15?\"\n\
\n\
assert = [\n\
  { type = \"contains\",  value = \"42\", weight = 3.0 },\n\
  { type = \"llm_judge\", rubric = \"Answer must state 42 events on 2026-01-15. Must not add facts not in context.\", threshold = 0.85, weight = 2.0 },\n\
]\n\
\n\
[[tests]]\n\
name        = \"rag-no-hallucination\"\n\
description = \"Must decline when context does not contain the answer\"\n\
context     = [\"The system processed 42 events on 2026-01-15.\"]\n\
prompt      = \"What was the revenue in Q3 2025?\"\n\
\n\
assert = [\n\
  { type = \"llm_judge\", rubric = \"Response must acknowledge it cannot answer from the provided context. Must not invent a revenue figure.\", threshold = 0.85, weight = 3.0 },\n\
  { type = \"not_contains\", value = \"$\", weight = 1.0 },\n\
]\n";

// ── Strategy-specific test blocks ─────────────────────────────────────────────

fn rag_chunk_boundary_test() -> &'static str {
    "\n\
[[tests]]\n\
name        = \"rag-chunk-boundary\"\n\
description = \"Answer requires info from two adjacent chunks — retriever must fetch both\"\n\
context     = [\n\
  \"Q3 revenue was $4.2M, up from $3.56M in Q2.\",\n\
  \"The 18% growth was driven by enterprise sales closing in September.\",\n\
]\n\
prompt      = \"By what percentage did revenue grow from Q2 to Q3, and what drove the growth?\"\n\
\n\
assert = [\n\
  { type = \"contains\",  value = \"18\",         weight = 2.0 },\n\
  { type = \"contains\",  value = \"enterprise\",  weight = 2.0 },\n\
  { type = \"llm_judge\", rubric = \"Must include BOTH the 18% growth figure AND the enterprise sales driver. Missing either indicates the retriever fetched only one chunk — a chunk-boundary failure.\", threshold = 0.80, weight = 3.0 },\n\
]\n"
}

fn rag_reranker_sensitivity_test() -> &'static str {
    "\n\
[[tests]]\n\
name        = \"rag-reranker-demotes-lexical-match\"\n\
description = \"Lexically similar but semantically wrong doc must be demoted by reranker\"\n\
context     = [\n\
  \"Python is a high-level programming language valued for readability and simplicity.\",\n\
  \"Python (Pythonidae) is a family of large nonvenomous snakes found in Africa and Asia.\",\n\
]\n\
prompt      = \"What programming language is known for readable and simple syntax?\"\n\
\n\
assert = [\n\
  { type = \"contains\",     value = \"programming\", weight = 2.0 },\n\
  { type = \"not_contains\", value = \"snake\",        weight = 3.0 },\n\
  { type = \"not_contains\", value = \"Africa\",       weight = 3.0 },\n\
  { type = \"llm_judge\", rubric = \"Must describe the Python programming language. Any mention of snakes or Africa means the reranker failed to demote the wrong document.\", threshold = 0.90, weight = 3.0 },\n\
]\n"
}

fn rag_hybrid_keyword_test() -> &'static str {
    "\n\
[[tests]]\n\
name        = \"rag-hybrid-keyword-exact\"\n\
description = \"Exact identifier query — BM25 component of hybrid search must retrieve the right doc\"\n\
context     = [\n\
  \"Product SKU-4872-X carries a 5-year warranty from date of purchase.\",\n\
  \"Standard products include a 1-year limited warranty.\",\n\
]\n\
prompt      = \"What is the warranty period for SKU-4872-X?\"\n\
\n\
assert = [\n\
  { type = \"contains\",     value = \"5\",      weight = 3.0 },\n\
  { type = \"not_contains\", value = \"1-year\",  weight = 2.0 },\n\
  { type = \"llm_judge\", rubric = \"Must state the 5-year warranty for SKU-4872-X. Answering 1 year means BM25 failed on the exact product code.\", threshold = 0.85, weight = 3.0 },\n\
]\n"
}

fn rag_hybrid_semantic_test() -> &'static str {
    "\n\
[[tests]]\n\
name        = \"rag-hybrid-semantic-paraphrase\"\n\
description = \"No shared keywords with source — dense vector side of hybrid must handle this\"\n\
context     = [\n\
  \"Our refund policy allows customers to return items within 30 days for a full reimbursement.\",\n\
]\n\
prompt      = \"How long do I have to send something back if I change my mind?\"\n\
\n\
assert = [\n\
  { type = \"contains\",  value = \"30\", weight = 3.0 },\n\
  { type = \"llm_judge\", rubric = \"Must state 30 days. The query shares no keywords with the source (no refund, return, policy) — only the dense vector component can bridge this gap.\", threshold = 0.85, weight = 3.0 },\n\
]\n"
}

fn rag_hyde_test() -> &'static str {
    "\n\
[[tests]]\n\
name        = \"rag-hyde-keywordless-query\"\n\
description = \"Abstract question with no source keywords — HyDE must generate a hypothesis to bridge it\"\n\
context     = [\n\
  \"Quarterly earnings exceeded analyst expectations by 23%, driven by strong enterprise sales.\",\n\
]\n\
prompt      = \"Were the financial results better or worse than anticipated?\"\n\
\n\
assert = [\n\
  { type = \"llm_judge\", rubric = \"Must say results exceeded or beat expectations. The query shares no vocabulary with the source — only HyDE can bridge this gap via a hypothetical document.\", threshold = 0.85, weight = 3.0 },\n\
]\n"
}

fn rag_multi_query_test() -> &'static str {
    "\n\
[[tests]]\n\
name        = \"rag-multi-query-ambiguous-intent\"\n\
description = \"Ambiguous query needs multiple retrieval angles — multi-query must generate variants\"\n\
context     = [\n\
  \"Mercury is the smallest and fastest planet, orbiting the sun in 88 days.\",\n\
  \"Mercury (Hg, atomic number 80) is a dense silvery liquid metal used in thermometers.\",\n\
  \"Freddie Mercury was the lead vocalist of Queen, famous for Bohemian Rhapsody.\",\n\
]\n\
prompt      = \"Tell me about Mercury.\"\n\
\n\
assert = [\n\
  { type = \"llm_judge\", rubric = \"Response should cover at least two of: the planet, the element, or the musician. Covering only one meaning indicates multi-query retrieval generated only a single query variant.\", threshold = 0.75, weight = 3.0 },\n\
]\n"
}

fn rag_late_chunking_test() -> &'static str {
    "\n\
[[tests]]\n\
name        = \"rag-late-chunking-section-retention\"\n\
description = \"Answer buried in a specific section — late chunking must retain full-doc context\"\n\
context     = [\n\
  \"Section 1: Introduction. This guide covers the REST API.\",\n\
  \"Section 2: Authentication. Use Bearer tokens for all requests.\",\n\
  \"Section 3: Rate Limits. Free tier: 100 req/min. Pro tier: 10,000 req/min.\",\n\
  \"Section 4: Endpoints. GET /v1/models returns the model list.\",\n\
  \"Section 5: Pricing. Pro tier is $49/month. Enterprise: contact sales.\",\n\
]\n\
prompt      = \"What are the rate limits for the Pro tier?\"\n\
\n\
assert = [\n\
  { type = \"contains\",     value = \"10,000\",  weight = 3.0 },\n\
  { type = \"not_contains\", value = \"100 req\",  weight = 2.0 },\n\
  { type = \"llm_judge\", rubric = \"Must state 10,000 req/min for Pro tier. Returning 100 req/min means the wrong section was retrieved — late chunking should preserve document structure.\", threshold = 0.85, weight = 3.0 },\n\
]\n"
}

fn rag_parent_doc_test() -> &'static str {
    "\n\
[[tests]]\n\
name        = \"rag-parent-doc-small-to-big\"\n\
description = \"Small chunk matches query but full answer requires the parent document\"\n\
context     = [\n\
  \"CEO statement: This acquisition is valued at $2.3B, closing Q3 2025. Full merger completes by end of 2025.\",\n\
]\n\
prompt      = \"When does the full merger complete?\"\n\
\n\
assert = [\n\
  { type = \"contains\",  value = \"2025\",  weight = 2.0 },\n\
  { type = \"contains\",  value = \"end\",   weight = 2.0 },\n\
  { type = \"llm_judge\", rubric = \"Must state the merger completes by end of 2025. Tests that parent-doc retrieval fetches the full statement, not just the $2.3B child chunk that matches lexically.\", threshold = 0.85, weight = 3.0 },\n\
]\n"
}

fn rag_contextual_compression_test() -> &'static str {
    "\n\
[[tests]]\n\
name        = \"rag-contextual-compression-noise\"\n\
description = \"Retrieved document is 90% noise — compressor must extract only the relevant span\"\n\
context     = [\n\
  \"Founded 2010. 500 employees. HQ in San Francisco. B2B SaaS. Revenue $45M. Series C. The API supports 12 languages. ISO 27001 certified.\",\n\
]\n\
prompt      = \"How many languages does the API support?\"\n\
\n\
assert = [\n\
  { type = \"contains\",  value = \"12\", weight = 3.0 },\n\
  { type = \"llm_judge\", rubric = \"Must answer 12 languages. Context has many irrelevant facts. If confused by noise, contextual compression failed.\", threshold = 0.85, weight = 3.0 },\n\
]\n"
}

// ── AI service suite ──────────────────────────────────────────────────────────

fn generate_ai_service_suite(
    finding: &Finding,
    provider: &str,
    out_dir: &Path,
    idx: usize,
) -> Result<(String, String)> {
    let basename = PathBuf::from(&finding.path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("service")
        .to_string();

    let suite_name = format!("Auto — {basename} ({provider} service)");

    let content = format!(
        r#"# AUTO-GENERATED by `crucible autodiscover`
# Source: {source}
# Provider: {provider}
# NOTE: this suite requires a live {provider} connection.
# Add your test cases below. Examples provided as starting points.

[suite]
name                 = "{suite_name}"
model                = "llama3:latest"
judge                = "llama3:latest"
category             = "autodiscover"
concurrency          = 2
n_runs               = 1
regression_threshold = 0.10

[[tests]]
name        = "basic-factual-response"
description = "Model should answer a simple factual question correctly"
prompt      = "Answer with one word: what is 2 + 2?"

assert = [
  {{ type = "contains", value = "4", weight = 3.0 }},
]

[[tests]]
name        = "no-hallucination-unknown"
description = "Model should refuse to answer unknown/private data questions"
prompt      = "What is the exact revenue for vendor ABC123 last month?"

assert = [
  {{ type = "llm_judge", rubric = "Response must not invent a specific number. Should say it doesn't know or ask for clarification.", threshold = 0.80, weight = 3.0 }},
]
"#,
        source = finding.path,
    );

    let suite_path = out_dir.join(format!("discover_{idx:02}_{basename}_service.toml"));
    fs::write(&suite_path, content)?;
    Ok((suite_path.display().to_string(), suite_name))
}

// ── Shim templates ────────────────────────────────────────────────────────────

/// Generate a TypeScript shim for any NL2SQL validator function.
/// The shim is generic: it imports the function from the detected file path.
fn nl2sql_shim_content(function_name: &str, source_file: &str) -> String {
    format!(
        r#"/**
 * AUTO-GENERATED by `crucible autodiscover`
 * NL2SQL security shim — wraps {function_name} for Crucible testing.
 *
 * Input  (stdin):  {{ sql, vendorId, params?, expectReject?, checkNotContains?, checkContains? }}
 * Output (stdout): {{ pass: boolean, reason?: string, sql?: string }}
 */
import * as fs from 'fs';
// Dynamic import from the detected source location
const {{ {function_name} }} = require('{source_file}');

interface ShimInput {{
  sql:               string;
  vendorId:          string;
  params?:           Array<{{ name: string; value: string | number }}>;
  expectReject?:     boolean;
  checkNotContains?: string;
  checkContains?:    string;
}}

async function main(): Promise<void> {{
  const raw = fs.readFileSync(0, 'utf8').trim();
  let input: ShimInput;
  try {{ input = JSON.parse(raw); }}
  catch (e) {{
    process.stdout.write(JSON.stringify({{ pass: false, reason: `Invalid JSON: ${{(e as Error).message}}` }}) + '\n');
    process.exit(1);
  }}

  const {{ sql, vendorId, params, expectReject = false, checkNotContains, checkContains }} = input;

  try {{
    const result = await {function_name}(sql, vendorId, params);
    const sanitized: string = typeof result === 'string' ? result : result.sql ?? JSON.stringify(result);

    if (expectReject) {{
      process.stdout.write(JSON.stringify({{ pass: false, reason: 'Expected rejection but query was accepted', sql: sanitized }}) + '\n');
      return;
    }}
    if (checkNotContains && sanitized.includes(checkNotContains)) {{
      process.stdout.write(JSON.stringify({{ pass: false, reason: `Sanitized SQL still contains: "${{checkNotContains}}"`, sql: sanitized }}) + '\n');
      return;
    }}
    if (checkContains && !sanitized.includes(checkContains)) {{
      process.stdout.write(JSON.stringify({{ pass: false, reason: `Sanitized SQL missing: "${{checkContains}}"`, sql: sanitized }}) + '\n');
      return;
    }}
    process.stdout.write(JSON.stringify({{ pass: true, sql: sanitized }}) + '\n');
  }} catch (err) {{
    const msg = (err as Error).message ?? String(err);
    if (expectReject) {{
      process.stdout.write(JSON.stringify({{ pass: true, reason: msg }}) + '\n');
    }} else {{
      process.stdout.write(JSON.stringify({{ pass: false, reason: `Unexpected rejection: ${{msg}}` }}) + '\n');
    }}
  }}
}}

main().catch(err => {{
  process.stdout.write(JSON.stringify({{ pass: false, reason: `Shim crashed: ${{(err as Error).message}}` }}) + '\n');
  process.exit(1);
}});
"#,
        function_name = function_name,
        source_file = source_file,
    )
}

fn python_nl2sql_shim_content(function_name: &str, source_file: &str) -> String {
    format!(
        r#"#!/usr/bin/env python3
# AUTO-GENERATED by `crucible autodiscover`
import sys, json, importlib.util

spec = importlib.util.spec_from_file_location("nl2sql_module", "{source_file}")
mod  = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
fn = getattr(mod, "{function_name}")

raw = sys.stdin.read().strip()
try:
    inp = json.loads(raw)
except json.JSONDecodeError as e:
    print(json.dumps({{"pass": False, "reason": f"Invalid JSON: {{e}}"}}))
    sys.exit(1)

sql         = inp["sql"]
vendor_id   = inp.get("vendorId", inp.get("vendor_id", "test_vendor"))
expect_rej  = inp.get("expectReject", inp.get("expect_reject", False))

try:
    result = fn(sql, vendor_id)
    if expect_rej:
        print(json.dumps({{"pass": False, "reason": "Expected rejection but query was accepted"}}))
    else:
        out = result if isinstance(result, str) else json.dumps(result)
        print(json.dumps({{"pass": True, "sql": out}}))
except Exception as e:
    msg = str(e)
    if expect_rej:
        print(json.dumps({{"pass": True, "reason": msg}}))
    else:
        print(json.dumps({{"pass": False, "reason": f"Unexpected rejection: {{msg}}"}}))
"#,
        function_name = function_name,
        source_file = source_file,
    )
}

// ── MCP server suite ──────────────────────────────────────────────────────────
//
// Generates an HTTP-provider test suite targeting the MCP server's HTTP
// endpoints (SSE and/or Streamable HTTP transports).
// Includes: initialize handshake, tools/list, tools/call (ping), error handling.

#[allow(clippy::too_many_arguments)]
fn generate_mcp_suite(
    finding: &Finding,
    framework: &McpFramework,
    transports: &[McpTransport],
    tool_count: usize,
    server_name: Option<&str>,
    out_dir: &Path,
    idx: usize,
) -> Result<(String, String)> {
    let suite_path = out_dir.join(format!("{idx:02}_mcp_server.toml"));

    let fw_label = match framework {
        McpFramework::ReKogMcpNest => "@rekog/mcp-nest (NestJS)",
        McpFramework::RawSdk => "@modelcontextprotocol/sdk",
        McpFramework::FastMcp => "fastmcp",
        McpFramework::Unknown => "unknown",
    };

    let name_comment = server_name
        .map(|n| format!("# Server name: {n}\n"))
        .unwrap_or_default();

    let transport_comment = transports
        .iter()
        .map(|t| t.to_string())
        .collect::<Vec<_>>()
        .join(", ");

    // Streamable HTTP preferred (2025 spec); fall back to SSE.
    let (mcp_path, mcp_note) = if transports.contains(&McpTransport::StreamableHttp) {
        ("/mcp", "Streamable HTTP endpoint (MCP 2025 spec)")
    } else if transports.contains(&McpTransport::Sse) {
        ("/sse", "SSE endpoint (legacy MCP transport)")
    } else {
        (
            "/mcp",
            "stdio-only — HTTP tests require a transport wrapper",
        )
    };

    let has_http = transports.contains(&McpTransport::Sse)
        || transports.contains(&McpTransport::StreamableHttp);

    let init_body = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"crucible","version":"1.0"}}}"#;
    let list_body = r#"{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}"#;
    let ping_body =
        r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"ping","arguments":{}}}"#;

    let http_tests = if has_http {
        format!(
            r#"
# ── Test 1: initialize handshake ─────────────────────────────────────────────
[[tests]]
name        = "mcp-initialize"
description = "MCP initialize handshake returns protocolVersion"
method      = "POST"
path        = "{mcp_path}"
body        = '{init_body}'
assert      = [
    {{ type = "http_status",   code = 200 }},
    {{ type = "json_field",    path = "$.result.protocolVersion" }},
    {{ type = "latency_under", ms = 5000  }},
]

# ── Test 2: tools/list ────────────────────────────────────────────────────────
[[tests]]
name        = "mcp-tools-list"
description = "tools/list returns tool registry ({tool_count} tools expected)"
method      = "POST"
path        = "{mcp_path}"
body        = '{list_body}'
assert      = [
    {{ type = "http_status",   code = 200              }},
    {{ type = "json_field",    path = "$.result.tools"  }},
    {{ type = "json_field",    path = "$.id", equals = "2" }},
    {{ type = "latency_under", ms = 3000               }},
]

# ── Test 3: ping tool call ────────────────────────────────────────────────────
[[tests]]
name        = "mcp-tool-ping"
description = "Call the ping tool — verifies tool routing is working"
method      = "POST"
path        = "{mcp_path}"
body        = '{ping_body}'
assert      = [
    {{ type = "http_status",   code = 200  }},
    {{ type = "json_field",    path = "$.result.content[0].type", equals = "text" }},
    {{ type = "latency_under", ms = 3000   }},
]

# ── Test 4: unknown method returns JSON-RPC error ─────────────────────────────
[[tests]]
name        = "mcp-unknown-method"
description = "Unknown method returns JSON-RPC error object (not HTTP 4xx)"
method      = "POST"
path        = "{mcp_path}"
body        = '{{"jsonrpc":"2.0","id":9,"method":"not/real","params":{{}}}}'
assert      = [
    {{ type = "http_status", code = 200       }},
    {{ type = "json_field",  path = "$.error" }},
]
"#,
            mcp_path = mcp_path,
            tool_count = tool_count,
            init_body = init_body,
            list_body = list_body,
            ping_body = ping_body,
        )
    } else {
        "\n# Stdio-only MCP server — no HTTP endpoints to test.\n\
         # Use the script provider with a shim that spawns the process.\n"
            .to_string()
    };

    let content = format!(
        r#"# ── Auto-generated MCP server test suite ─────────────────────────────────────
# Detected: {fw_label}
{name_comment}# Transport(s): {transport_comment}
# Tools found: {tool_count}
# Source:      {source}
# Note:        {mcp_note}
#
# Run: crucible run -s <this_file>
# Requires the MCP server running at base_url below.

[suite]
name        = "MCP Server Tests"
model       = "mock"
judge       = "llama3.1:8b"
concurrency = 2

[suite.http]
base_url   = "http://localhost:3000"   # ← update to your server URL
timeout_ms = 10000

[suite.http.headers]
Content-Type = "application/json"
# Authorization = "Bearer <token>"
{http_tests}"#,
        fw_label = fw_label,
        name_comment = name_comment,
        transport_comment = transport_comment,
        tool_count = tool_count,
        source = finding.path,
        mcp_note = mcp_note,
        http_tests = http_tests,
    );

    fs::write(&suite_path, &content)?;
    let desc = format!("MCP server suite ({transport_comment}) — {tool_count} tools");
    Ok((suite_path.display().to_string(), desc))
}
