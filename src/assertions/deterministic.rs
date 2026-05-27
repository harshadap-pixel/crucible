use crate::assertions::AssertionResult;
use anyhow::Result;
use regex::Regex;
use serde_json::Value;
use std::path::Path;

pub fn check_regex(pattern: &str, output: &str, weight: f64) -> Result<AssertionResult> {
    let re = Regex::new(pattern)?;
    let passed = re.is_match(output);
    Ok(AssertionResult {
        kind: "regex".into(),
        passed,
        score: if passed { 1.0 } else { 0.0 },
        reason: if passed {
            format!("Pattern /{pattern}/ matched")
        } else {
            format!("Pattern /{pattern}/ did not match")
        },
        weight,
    })
}

/// Navigate a dot/bracket JSON path and assert on the value.
/// Supported path syntax:
///   $.field          → object key
///   $.a.b.c          → nested keys
///   $[0]             → array index
///   $.items[2].name  → mixed
pub fn check_json_field(
    path: &str,
    equals: Option<&str>,
    contains: Option<&str>,
    output: &str,
    weight: f64,
) -> Result<AssertionResult> {
    // 1. Parse output as JSON
    let root: Value = match serde_json::from_str(output.trim()) {
        Ok(v) => v,
        Err(e) => {
            return Ok(AssertionResult {
                kind: "json_field".into(),
                passed: false,
                score: 0.0,
                reason: format!("Output is not valid JSON: {e}"),
                weight,
            })
        }
    };

    // 2. Navigate path
    let node = navigate_json_path(&root, path);
    let node = match node {
        Some(n) => n,
        None => {
            return Ok(AssertionResult {
                kind: "json_field".into(),
                passed: false,
                score: 0.0,
                reason: format!("Path '{path}' not found in JSON output"),
                weight,
            })
        }
    };

    // 3. Serialize node to string for comparison
    let node_str = match node {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    };

    // 4. Apply operator
    let (passed, reason) = if let Some(expected) = equals {
        let ok = node_str == expected;
        (
            ok,
            if ok {
                format!("'{path}' = '{expected}'")
            } else {
                format!("'{path}' expected '{expected}', got '{node_str}'")
            },
        )
    } else if let Some(substr) = contains {
        let ok = node_str.contains(substr);
        (
            ok,
            if ok {
                format!("'{path}' contains '{substr}'")
            } else {
                format!("'{path}' (= '{node_str}') does not contain '{substr}'")
            },
        )
    } else {
        // Just checking the field exists
        (true, format!("'{path}' exists: '{node_str}'"))
    };

    Ok(AssertionResult {
        kind: "json_field".into(),
        passed,
        score: if passed { 1.0 } else { 0.0 },
        reason,
        weight,
    })
}

/// Walk a JSON value following a simple path: $.a.b[0].c
fn navigate_json_path<'a>(root: &'a Value, path: &str) -> Option<&'a Value> {
    // Strip leading "$" or "$."
    let path = path
        .strip_prefix("$.")
        .unwrap_or_else(|| path.strip_prefix('$').unwrap_or(path));

    let mut current = root;
    for segment in tokenize_path(path) {
        current = match segment {
            PathSegment::Key(k) => current.get(k)?,
            PathSegment::Index(i) => current.get(i)?,
        };
    }
    Some(current)
}

enum PathSegment<'a> {
    Key(&'a str),
    Index(usize),
}

fn tokenize_path(path: &str) -> Vec<PathSegment<'_>> {
    let mut segments = Vec::new();
    let mut rest = path;

    while !rest.is_empty() {
        if let Some(after_bracket) = rest.strip_prefix('[') {
            // Array index: [N]
            if let Some(end) = after_bracket.find(']') {
                let idx_str = &after_bracket[..end];
                if let Ok(idx) = idx_str.parse::<usize>() {
                    segments.push(PathSegment::Index(idx));
                }
                rest = &after_bracket[end + 1..];
                rest = rest.strip_prefix('.').unwrap_or(rest);
                continue;
            }
        }
        // Object key: up to next '.' or '['
        let end = rest.find(['.', '[']).unwrap_or(rest.len());
        segments.push(PathSegment::Key(&rest[..end]));
        rest = &rest[end..];
        rest = rest.strip_prefix('.').unwrap_or(rest);
    }
    segments
}

pub fn check_json_schema(schema_str: &str, output: &str, weight: f64) -> Result<AssertionResult> {
    // Parse output as JSON first
    let instance = match serde_json::from_str::<serde_json::Value>(output.trim()) {
        Ok(v) => v,
        Err(e) => {
            return Ok(AssertionResult {
                kind: "json_schema".into(),
                passed: false,
                score: 0.0,
                reason: format!("Output is not valid JSON: {e}"),
                weight,
            });
        }
    };

    let schema = match serde_json::from_str::<serde_json::Value>(schema_str) {
        Ok(v) => v,
        Err(e) => {
            return Ok(AssertionResult {
                kind: "json_schema".into(),
                passed: false,
                score: 0.0,
                reason: format!("Suite schema definition is invalid JSON: {e}"),
                weight,
            });
        }
    };

    let compiled = jsonschema::validator_for(&schema)
        .map_err(|e| anyhow::anyhow!("Schema compile error: {e}"))?;

    let errors: Vec<String> = compiled
        .iter_errors(&instance)
        .map(|e| e.to_string())
        .collect();

    let passed = errors.is_empty();
    Ok(AssertionResult {
        kind: "json_schema".into(),
        passed,
        score: if passed { 1.0 } else { 0.0 },
        reason: if passed {
            "JSON schema valid".into()
        } else {
            format!("Schema violations: {}", errors.join("; "))
        },
        weight,
    })
}

pub fn check_contains(value: &str, output: &str, weight: f64) -> AssertionResult {
    let passed = output.contains(value);
    AssertionResult {
        kind: "contains".into(),
        passed,
        score: if passed { 1.0 } else { 0.0 },
        reason: if passed {
            format!("Output contains '{value}'")
        } else {
            format!("Output does not contain '{value}'")
        },
        weight,
    }
}

pub fn check_not_contains(value: &str, output: &str, weight: f64) -> AssertionResult {
    let passed = !output.contains(value);
    AssertionResult {
        kind: "not_contains".into(),
        passed,
        score: if passed { 1.0 } else { 0.0 },
        reason: if passed {
            format!("Output does not contain '{value}' ✓")
        } else {
            format!("Output unexpectedly contains '{value}'")
        },
        weight,
    }
}

/// Compare `output` against a golden file on disk.
///
/// Behaviour:
///   • File absent  → write output as new golden, always PASS  (first-run creation)
///   • File present → PASS if trimmed content matches, FAIL otherwise
///   • `update`     → overwrite existing golden with current output, always PASS
///
/// The `suite_dir` is the directory of the suite file, used to resolve relative paths.
pub fn check_snapshot(
    file: &str,
    output: &str,
    weight: f64,
    suite_dir: &str,
    update: bool,
) -> Result<AssertionResult> {
    // Resolve path: absolute as-is, relative joined to suite directory
    let path = if Path::new(file).is_absolute() {
        Path::new(file).to_path_buf()
    } else {
        Path::new(suite_dir).join(file)
    };

    // Create parent directories if needed
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let trimmed = output.trim();

    // Update mode: overwrite and pass
    if update {
        std::fs::write(&path, trimmed)?;
        return Ok(AssertionResult {
            kind: "snapshot".into(),
            passed: true,
            score: 1.0,
            reason: format!("Snapshot updated: {}", path.display()),
            weight,
        });
    }

    // First run: golden doesn't exist yet — write it and pass
    if !path.exists() {
        std::fs::write(&path, trimmed)?;
        return Ok(AssertionResult {
            kind: "snapshot".into(),
            passed: true,
            score: 1.0,
            reason: format!("Snapshot created: {}", path.display()),
            weight,
        });
    }

    // Compare against existing golden
    let golden = std::fs::read_to_string(&path)?;
    let golden = golden.trim();
    let passed = trimmed == golden;

    Ok(AssertionResult {
        kind: "snapshot".into(),
        passed,
        score: if passed { 1.0 } else { 0.0 },
        reason: if passed {
            format!("Matches golden: {}", path.display())
        } else {
            // Show a short diff hint (first differing line)
            let first_diff = trimmed
                .lines()
                .zip(golden.lines())
                .enumerate()
                .find(|(_, (a, b))| a != b)
                .map(|(i, (got, exp))| {
                    format!(" line {}: got {:?}, expected {:?}", i + 1, got, exp)
                })
                .unwrap_or_else(|| format!(" length {} vs {}", trimmed.len(), golden.len()));
            format!(
                "Snapshot mismatch ({}): {}{first_diff}",
                path.display(),
                if trimmed.len() != golden.len() {
                    "length differs —"
                } else {
                    ""
                }
            )
        },
        weight,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── check_contains ────────────────────────────────────────────────────────

    #[test]
    fn contains_passes_when_substring_present() {
        let r = check_contains("hello", "hello world", 1.0);
        assert!(r.passed);
        assert_eq!(r.score, 1.0);
    }

    #[test]
    fn contains_fails_when_substring_absent() {
        let r = check_contains("missing", "hello world", 1.0);
        assert!(!r.passed);
        assert_eq!(r.score, 0.0);
    }

    #[test]
    fn contains_is_case_sensitive() {
        let r = check_contains("Hello", "hello world", 1.0);
        assert!(!r.passed, "contains should be case-sensitive");
    }

    // ── check_not_contains ────────────────────────────────────────────────────

    #[test]
    fn not_contains_passes_when_absent() {
        let r = check_not_contains("error", "everything is fine", 1.0);
        assert!(r.passed);
        assert_eq!(r.score, 1.0);
    }

    #[test]
    fn not_contains_fails_when_present() {
        let r = check_not_contains("error", "fatal error occurred", 1.0);
        assert!(!r.passed);
        assert_eq!(r.score, 0.0);
    }

    // ── check_regex ───────────────────────────────────────────────────────────

    #[test]
    fn regex_matches_pattern() {
        let r = check_regex(r"\d{4}", "year: 2026", 1.0).unwrap();
        assert!(r.passed);
    }

    #[test]
    fn regex_fails_when_no_match() {
        let r = check_regex(r"^\d+$", "not a number", 1.0).unwrap();
        assert!(!r.passed);
    }

    #[test]
    fn regex_errors_on_invalid_pattern() {
        let result = check_regex(r"[invalid", "text", 1.0);
        assert!(result.is_err());
    }

    // ── check_json_field ──────────────────────────────────────────────────────

    #[test]
    fn json_field_exists_passes() {
        let json = r#"{"status":"ok","count":3}"#;
        let r = check_json_field("$.status", None, None, json, 1.0).unwrap();
        assert!(r.passed);
    }

    #[test]
    fn json_field_equals_passes() {
        let json = r#"{"status":"ok"}"#;
        let r = check_json_field("$.status", Some("ok"), None, json, 1.0).unwrap();
        assert!(r.passed);
    }

    #[test]
    fn json_field_equals_fails_on_mismatch() {
        let json = r#"{"status":"error"}"#;
        let r = check_json_field("$.status", Some("ok"), None, json, 1.0).unwrap();
        assert!(!r.passed);
    }

    #[test]
    fn json_field_missing_path_fails() {
        let json = r#"{"a":1}"#;
        let r = check_json_field("$.b.c", None, None, json, 1.0).unwrap();
        assert!(!r.passed);
    }

    #[test]
    fn json_field_array_index() {
        let json = r#"{"items":["x","y","z"]}"#;
        let r = check_json_field("$.items[1]", Some("y"), None, json, 1.0).unwrap();
        assert!(r.passed);
    }

    #[test]
    fn json_field_invalid_json_fails_gracefully() {
        let r = check_json_field("$.x", None, None, "not json", 1.0).unwrap();
        assert!(!r.passed);
        assert!(r.reason.contains("not valid JSON"));
    }

    // ── check_json_schema ─────────────────────────────────────────────────────

    #[test]
    fn json_schema_valid_passes() {
        let schema =
            r#"{"type":"object","required":["name"],"properties":{"name":{"type":"string"}}}"#;
        let output = r#"{"name":"Alice"}"#;
        let r = check_json_schema(schema, output, 1.0).unwrap();
        assert!(r.passed);
    }

    #[test]
    fn json_schema_missing_required_fails() {
        let schema =
            r#"{"type":"object","required":["name"],"properties":{"name":{"type":"string"}}}"#;
        let output = r#"{"age":30}"#;
        let r = check_json_schema(schema, output, 1.0).unwrap();
        assert!(!r.passed);
    }
}
