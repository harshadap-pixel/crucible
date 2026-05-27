/// Script provider — runs an external process, sends prompt via stdin,
/// reads response from stdout. Lets Crucible test any local script
/// including evaluate.py, BPE pipelines, or any CLI tool.
use anyhow::{bail, Result};
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct ScriptResult {
    pub stdout: String,
    pub latency_ms: u128,
    #[allow(dead_code)]
    pub exit_code: i32,
}

/// Run `cmd` with `prompt` passed on stdin, capture stdout.
/// Extra env vars (e.g. TASK_TYPE=classification) can be injected.
pub fn run(cmd: &str, args: &[&str], prompt: &str, env: &[(&str, &str)]) -> Result<ScriptResult> {
    let t0 = Instant::now();

    let mut cmd_builder = Command::new(cmd);
    cmd_builder
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    for (k, v) in env {
        cmd_builder.env(k, v);
    }

    let mut child = cmd_builder.spawn()?;

    // Write prompt to stdin
    if let Some(stdin) = child.stdin.take() {
        let mut stdin = stdin;
        stdin.write_all(prompt.as_bytes())?;
        // stdin closes when it drops, signalling EOF to the child
    }

    let output = child.wait_with_output()?;
    let latency_ms = t0.elapsed().as_millis();

    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    let exit_code = output.status.code().unwrap_or(-1);

    if exit_code != 0 && stdout.is_empty() {
        bail!("Script exited with code {exit_code}: {stderr}");
    }

    Ok(ScriptResult {
        stdout,
        latency_ms,
        exit_code,
    })
}
