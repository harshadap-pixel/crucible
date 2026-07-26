/// Pattern detection abstraction layer
/// Supports both regex (fallback) and tree-sitter (primary) backends
use anyhow::Result;
use std::path::Path;

/// Result of detecting a specific code pattern
#[derive(Debug, Clone)]
pub struct PatternMatch {
    pub pattern_name: String,
    pub confidence: f64, // 0.0-1.0, higher is more certain
    pub signals: Vec<String>,
    pub details: String,
}

/// Language supported by tree-sitter
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Language {
    TypeScript,
    JavaScript,
    Python,
    Unknown,
}

impl Language {
    pub fn from_path(path: &str) -> Self {
        match Path::new(path).extension().and_then(|ext| ext.to_str()) {
            Some("ts") | Some("tsx") => Language::TypeScript,
            Some("js") | Some("jsx") => Language::JavaScript,
            Some("py") => Language::Python,
            _ => Language::Unknown,
        }
    }
}

/// Trait for pluggable pattern detection backends
pub trait PatternDetector: Send + Sync {
    /// Detect patterns in file content
    fn detect(&self, content: &str, language: Language, path: &str) -> Result<Vec<PatternMatch>>;

    /// Human-readable name of detector
    fn name(&self) -> &str;
}

/// Regex-based detector (fallback, always available)
pub struct RegexDetector;

impl PatternDetector for RegexDetector {
    fn detect(
        &self,
        _content: &str,
        _language: Language,
        _path: &str,
    ) -> Result<Vec<PatternMatch>> {
        // Placeholder: existing regex logic will go here
        Ok(Vec::new())
    }

    fn name(&self) -> &str {
        "regex"
    }
}

/// Tree-sitter AST-based detector (primary)
pub struct TreeSitterDetector {
    ts_language: tree_sitter::Language,
}

impl TreeSitterDetector {
    pub fn new(language: Language) -> Result<Self> {
        let ts_language = match language {
            Language::TypeScript | Language::JavaScript => {
                tree_sitter_typescript::language_typescript()
            }
            Language::Python => tree_sitter_python::language(),
            Language::Unknown => anyhow::bail!("Unsupported language for tree-sitter"),
        };

        Ok(Self { ts_language })
    }

    /// Query AST for specific patterns
    #[allow(dead_code)]
    fn query_pattern(
        &self,
        tree: &tree_sitter::Tree,
        query_str: &str,
        content: &[u8],
    ) -> Result<Vec<PatternMatch>> {
        let query = tree_sitter::Query::new(self.ts_language, query_str)?;
        let mut cursor = tree_sitter::QueryCursor::new();
        let matches = cursor.matches(&query, tree.root_node(), content);

        let mut results = Vec::new();
        for m in matches {
            if let Some(capture) = m.captures.first() {
                let line = capture.node.start_position().row + 1;
                results.push(PatternMatch {
                    pattern_name: "ast_match".to_string(),
                    confidence: 0.9, // AST matches are high-confidence
                    signals: vec![],
                    details: format!("Match at line {}", line),
                });
            }
        }

        Ok(results)
    }
}

impl PatternDetector for TreeSitterDetector {
    fn detect(&self, content: &str, language: Language, _path: &str) -> Result<Vec<PatternMatch>> {
        if language == Language::Unknown {
            return Ok(Vec::new());
        }

        let mut parser = tree_sitter::Parser::new();
        parser.set_language(self.ts_language)?;

        let _tree = parser
            .parse(content.as_bytes(), None)
            .ok_or_else(|| anyhow::anyhow!("Failed to parse code with tree-sitter"))?;

        // Placeholder: queries will be added based on patterns
        // For now, return empty to show structure is initialized
        Ok(Vec::new())
    }

    fn name(&self) -> &str {
        "tree-sitter"
    }
}

/// Dual-mode detector: tries tree-sitter first, falls back to regex
pub struct HybridDetector {
    tree_sitter: Option<TreeSitterDetector>,
    regex: RegexDetector,
}

impl HybridDetector {
    pub fn new(language: Language) -> Self {
        let tree_sitter = TreeSitterDetector::new(language).ok();
        let regex = RegexDetector;

        Self { tree_sitter, regex }
    }
}

impl PatternDetector for HybridDetector {
    fn detect(&self, content: &str, language: Language, path: &str) -> Result<Vec<PatternMatch>> {
        // Try tree-sitter first
        if let Some(ts) = &self.tree_sitter {
            if let Ok(matches) = ts.detect(content, language, path) {
                if !matches.is_empty() {
                    return Ok(matches);
                }
            }
        }

        // Fallback to regex
        self.regex.detect(content, language, path)
    }

    fn name(&self) -> &str {
        if self.tree_sitter.is_some() {
            "hybrid(tree-sitter+regex)"
        } else {
            "hybrid(regex)"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_language_detection_typescript() {
        assert_eq!(Language::from_path("index.ts"), Language::TypeScript);
        assert_eq!(Language::from_path("component.tsx"), Language::TypeScript);
    }

    #[test]
    fn test_language_detection_javascript() {
        assert_eq!(Language::from_path("index.js"), Language::JavaScript);
        assert_eq!(Language::from_path("component.jsx"), Language::JavaScript);
    }

    #[test]
    fn test_language_detection_python() {
        assert_eq!(Language::from_path("script.py"), Language::Python);
    }

    #[test]
    fn test_language_detection_unknown() {
        assert_eq!(Language::from_path("file.txt"), Language::Unknown);
        assert_eq!(Language::from_path("no_extension"), Language::Unknown);
    }

    #[test]
    fn test_regex_detector_available() {
        let detector = RegexDetector;
        assert_eq!(detector.name(), "regex");
    }

    #[test]
    fn test_hybrid_detector_fallback() {
        // Should work even if tree-sitter init fails
        let detector = HybridDetector::new(Language::Unknown);
        assert!(detector.name().contains("regex"));
    }
}
