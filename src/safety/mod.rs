use crate::config::Suite;
/// Safety probe library — OWASP LLM Top 10 + agent-specific
/// Probes are loaded from suites/safety/*.toml and executed via the standard runner.
/// This module provides probe-loading helpers and the refusal classifier.
use anyhow::Result;
use std::path::Path;

/// Load all *.toml suites from suites/safety/
#[allow(dead_code)]
pub fn load_all_safety_suites() -> Result<Vec<Suite>> {
    let dir = Path::new("suites/safety");
    if !dir.exists() {
        return Ok(vec![]);
    }
    let mut suites = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.extension().and_then(|e| e.to_str()) == Some("toml") {
            match crate::config::load(path.to_str().unwrap()) {
                Ok(s) => suites.push(s),
                Err(e) => eprintln!("  ⚠ Skipping {}: {e}", path.display()),
            }
        }
    }
    Ok(suites)
}
