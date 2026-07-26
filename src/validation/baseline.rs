/// Baseline management for regression tracking
/// Stores and compares judge validation results over time
use anyhow::Result;
use std::fs;
use std::path::PathBuf;

use super::judge_validator::{JudgeValidationBaseline, JudgeValidationReport};

const BASELINES_DIR: &str = ".crucible/baselines";

/// Manages baseline storage and retrieval
pub struct BaselineManager;

impl BaselineManager {
    /// Get the baselines directory path
    fn baselines_dir() -> Result<PathBuf> {
        let home = dirs::home_dir()
            .ok_or_else(|| anyhow::anyhow!("Could not determine home directory"))?;
        let path = home.join(BASELINES_DIR);
        fs::create_dir_all(&path)?;
        Ok(path)
    }

    /// Save a validation report as a new baseline
    pub fn save_baseline(report: &JudgeValidationReport, label: Option<String>) -> Result<PathBuf> {
        let baseline = JudgeValidationBaseline {
            timestamp: chrono::Local::now().to_rfc3339(),
            suite_name: report.suite_name.clone(),
            overall_agreement: report.overall_agreement,
            per_judge_agreement: Self::extract_judge_agreement(report),
            per_rubric_metrics: report.per_rubric_metrics.clone(),
        };

        let baselines_dir = Self::baselines_dir()?;

        // Generate filename: suite_name_timestamp.json
        let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S");
        let filename = if let Some(l) = label {
            format!("{}_{}.json", sanitize_filename(&report.suite_name), l)
        } else {
            format!(
                "{}_{}.json",
                sanitize_filename(&report.suite_name),
                timestamp
            )
        };

        let file_path = baselines_dir.join(&filename);

        // Write baseline to file
        let json = serde_json::to_string_pretty(&baseline)?;
        fs::write(&file_path, json)?;

        // Update CURRENT symlink
        let current_link = baselines_dir.join("CURRENT");
        if current_link.exists() {
            fs::remove_file(&current_link).ok();
        }

        #[cfg(unix)]
        std::os::unix::fs::symlink(&filename, &current_link)?;
        #[cfg(windows)]
        {
            // Windows requires copying instead of symlink
            fs::copy(&file_path, &current_link)?;
        }

        Ok(file_path)
    }

    /// Load the current baseline
    pub fn load_current_baseline() -> Result<Option<JudgeValidationBaseline>> {
        let baselines_dir = Self::baselines_dir()?;
        let current_link = baselines_dir.join("CURRENT");

        if !current_link.exists() {
            return Ok(None);
        }

        let content = fs::read_to_string(&current_link)?;
        let baseline = serde_json::from_str(&content)?;
        Ok(Some(baseline))
    }

    /// Load a specific baseline by name
    pub fn load_baseline(name: &str) -> Result<Option<JudgeValidationBaseline>> {
        let baselines_dir = Self::baselines_dir()?;
        let file_path = baselines_dir.join(format!("{}.json", name));

        if !file_path.exists() {
            return Ok(None);
        }

        let content = fs::read_to_string(file_path)?;
        let baseline = serde_json::from_str(&content)?;
        Ok(Some(baseline))
    }

    /// List all saved baselines
    pub fn list_baselines() -> Result<Vec<BaselineInfo>> {
        let baselines_dir = Self::baselines_dir()?;
        let mut baselines = Vec::new();

        for entry in fs::read_dir(&baselines_dir)? {
            let entry = entry?;
            let path = entry.path();

            // Skip non-JSON files and CURRENT link
            if path.extension().map(|e| e != "json").unwrap_or(true) {
                continue;
            }
            if path.file_name().map(|n| n == "CURRENT").unwrap_or(false) {
                continue;
            }

            if let Ok(content) = fs::read_to_string(&path) {
                if let Ok(baseline) = serde_json::from_str::<JudgeValidationBaseline>(&content) {
                    baselines.push(BaselineInfo {
                        filename: path.file_name().unwrap().to_string_lossy().to_string(),
                        suite_name: baseline.suite_name,
                        timestamp: baseline.timestamp,
                        overall_agreement: baseline.overall_agreement,
                    });
                }
            }
        }

        // Sort by timestamp descending (newest first)
        baselines.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        Ok(baselines)
    }

    /// Get info about the current baseline
    pub fn current_baseline_info() -> Result<Option<BaselineInfo>> {
        match Self::load_current_baseline()? {
            Some(baseline) => Ok(Some(BaselineInfo {
                filename: "CURRENT".to_string(),
                suite_name: baseline.suite_name,
                timestamp: baseline.timestamp,
                overall_agreement: baseline.overall_agreement,
            })),
            None => Ok(None),
        }
    }

    /// Extract per-judge agreement from report
    fn extract_judge_agreement(
        report: &JudgeValidationReport,
    ) -> std::collections::HashMap<String, f64> {
        // Simplified: use overall agreement for all judges
        // In future: track per-judge metrics separately
        report
            .judges_compared
            .iter()
            .map(|j| (j.clone(), report.overall_agreement))
            .collect()
    }
}

/// Information about a saved baseline
#[derive(Debug, Clone)]
pub struct BaselineInfo {
    pub filename: String,
    pub suite_name: String,
    pub timestamp: String,
    pub overall_agreement: f64,
}

/// Sanitize filename to be filesystem-safe
fn sanitize_filename(name: &str) -> String {
    name.replace('/', "_")
        .replace('\\', "_")
        .replace(':', "_")
        .replace('?', "_")
        .replace('*', "_")
        .replace('"', "_")
        .replace('<', "_")
        .replace('>', "_")
        .replace('|', "_")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize_filename() {
        assert_eq!(sanitize_filename("safety/owasp"), "safety_owasp");
        assert_eq!(sanitize_filename("test:suite"), "test_suite");
        assert_eq!(sanitize_filename("normal_name"), "normal_name");
    }

    #[test]
    fn test_baselines_dir_exists() {
        let result = BaselineManager::baselines_dir();
        assert!(result.is_ok());
        let dir = result.unwrap();
        assert!(dir.exists());
    }

    #[test]
    fn test_list_empty_baselines() {
        let result = BaselineManager::list_baselines();
        assert!(result.is_ok());
        // May or may not have baselines, just verify it doesn't error
    }

    #[test]
    fn test_no_current_baseline_initially() {
        let result = BaselineManager::load_current_baseline();
        assert!(result.is_ok());
        // May or may not exist initially
    }
}
