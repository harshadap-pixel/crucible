use rust_embed::Embed;
use std::path::{Path, PathBuf};

/// All `.toml` files under the `suites/` directory are baked into the binary.
#[derive(Embed)]
#[folder = "suites/"]
#[include = "*.toml"]
struct Suites;

/// Returns `~/.local/share/crucible` (cross-platform via the same logic as the DB).
pub fn data_dir() -> PathBuf {
    #[cfg(target_os = "macos")]
    let base = std::env::var("HOME")
        .ok()
        .map(|h| PathBuf::from(h).join("Library/Application Support"))
        .unwrap_or_else(|| PathBuf::from("."));

    #[cfg(target_os = "linux")]
    let base = std::env::var("XDG_DATA_HOME")
        .ok()
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::var("HOME")
                .ok()
                .map(|h| PathBuf::from(h).join(".local/share"))
                .unwrap_or_else(|| PathBuf::from("."))
        });

    #[cfg(target_os = "windows")]
    let base = std::env::var("APPDATA")
        .ok()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));

    base.join("crucible")
}

/// Returns the path to the installed suites directory, extracting embedded
/// suites on first call if the directory doesn't exist yet.
pub fn suites_dir() -> PathBuf {
    let dir = data_dir().join("suites");
    if !dir.exists() {
        extract_suites(&dir);
    }
    dir
}

/// Extract all embedded suite files to `dest`.
fn extract_suites(dest: &Path) {
    for name in Suites::iter() {
        let target = dest.join(name.as_ref());
        if let Some(parent) = target.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Some(file) = Suites::get(name.as_ref()) {
            let _ = std::fs::write(&target, file.data.as_ref());
        }
    }
}

/// Resolve a suite path: tries it as-is first (absolute or relative to CWD),
/// then falls back to the installed suites directory.
pub fn resolve_suite_path(path: &str) -> String {
    if std::path::Path::new(path).exists() {
        return path.to_string();
    }
    let installed = suites_dir().join(path);
    if installed.exists() {
        return installed.display().to_string();
    }
    // Return original — caller will surface the "file not found" error
    path.to_string()
}

/// Resolve a directory path for `--dir`, same logic as above.
pub fn resolve_dir_path(path: &str) -> String {
    if std::path::Path::new(path).exists() {
        return path.to_string();
    }
    let installed = suites_dir().join(path);
    if installed.exists() {
        return installed.display().to_string();
    }
    path.to_string()
}
