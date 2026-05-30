/// Self-update: downloads the latest release from GitHub and replaces the
/// current binary. Works on macOS (aarch64 + x86_64). Linux not yet supported
/// (no Linux release artifacts).
use anyhow::{bail, Result};
use colored::Colorize;

const REPO: &str = "harshadap-pixel/crucible";
const API_URL: &str = "https://api.github.com/repos/harshadap-pixel/crucible/releases/latest";

/// Detect the right asset name for the current platform.
fn asset_name() -> Result<&'static str> {
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;

    match (os, arch) {
        ("macos", "aarch64") => Ok("crucible-aarch64-macos"),
        ("macos", "x86_64") => Ok("crucible-x86_64-macos"),
        _ => bail!(
            "No pre-built binary available for {os}/{arch}.\n\
             Build from source: https://github.com/{REPO}"
        ),
    }
}

pub async fn run() -> Result<()> {
    let asset = asset_name()?;

    println!("\n{}", "CRUCIBLE UPDATE".bold().cyan());
    println!("{}", "─".repeat(62).dimmed());
    println!(
        "  Platform  {}/{}",
        std::env::consts::OS,
        std::env::consts::ARCH
    );
    println!("  Asset     {asset}");

    // ── Fetch latest release metadata ─────────────────────────────────────────
    print!("  Checking latest release... ");
    let client = reqwest::Client::builder()
        .user_agent("crucible-updater")
        .build()?;

    let release: serde_json::Value = client.get(API_URL).send().await?.json().await?;

    let tag = release["tag_name"]
        .as_str()
        .unwrap_or("unknown")
        .to_string();
    let current = env!("CARGO_PKG_VERSION");

    println!("{}", "done".green());
    println!("  Latest    {}", tag.yellow());
    println!("  Current   v{current}");

    if tag == format!("v{current}") {
        println!("\n  {} Already up to date.", "✓".green());
        return Ok(());
    }

    // ── Find download URL for our asset ──────────────────────────────────────
    let download_url = release["assets"]
        .as_array()
        .and_then(|assets| {
            assets
                .iter()
                .find(|a| a["name"].as_str().unwrap_or("") == asset)
        })
        .and_then(|a| a["browser_download_url"].as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| anyhow::anyhow!("Asset '{asset}' not found in release {tag}"))?;

    // ── Download ──────────────────────────────────────────────────────────────
    print!("  Downloading {tag}... ");
    let bytes = client.get(&download_url).send().await?.bytes().await?;
    println!("{} ({} KB)", "done".green(), bytes.len() / 1024);

    // ── Replace current binary atomically ────────────────────────────────────
    let current_exe = std::env::current_exe()?;
    let tmp_path = current_exe.with_extension("tmp");

    std::fs::write(&tmp_path, &bytes)?;

    // Make executable (unix only — on macOS/Linux)
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&tmp_path)?.permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&tmp_path, perms)?;
    }

    // Atomic replace: rename tmp over current binary
    std::fs::rename(&tmp_path, &current_exe)?;

    println!(
        "\n  {} Updated to {} → run `crucible --version` to confirm.",
        "✓".green().bold(),
        tag.yellow().bold()
    );
    println!("  Binary: {}", current_exe.display().to_string().dimmed());
    println!();

    Ok(())
}
