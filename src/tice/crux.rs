//! Crux Extractor — Find the one fact that collapses branches
//!
//! A crux must split the world into at least two materially different
//! downstream actions.
//!
//! Format: "If X then do A, if ¬X then do B"

use super::graph::{ClaimId, ConstraintGraph};

/// How to test a crux
#[derive(Debug, Clone)]
pub enum TestType {
    /// Run code, check exit status
    CodeExecution(String),
    /// Check if file/pattern exists
    FileCheck(String),
    /// Look up a fact
    Lookup(String),
    /// Requires human verification
    Manual,
}

/// A crux: the discriminating test
#[derive(Debug, Clone)]
pub struct Crux {
    /// The claim this crux tests
    pub target: ClaimId,
    /// The falsifiable statement
    pub falsifier: String,
    /// How to test it
    pub test_type: TestType,
    /// What happens if true
    pub if_true: String,
    /// What happens if false
    pub if_false: String,
    /// Expected outcome (for prediction)
    pub expected: Option<bool>,
}

/// Extracts cruxes from claims
pub struct CruxExtractor {
    /// Templates for common crux patterns
    #[allow(dead_code)]
    templates: Vec<CruxTemplate>,
}

#[allow(dead_code)]
struct CruxTemplate {
    pattern: &'static str,
    test_type_fn: fn(&str) -> TestType,
}

impl CruxExtractor {
    pub fn new() -> Self {
        Self {
            templates: vec![
                CruxTemplate {
                    pattern: "file exists",
                    test_type_fn: |s| TestType::FileCheck(s.to_string()),
                },
                CruxTemplate {
                    pattern: "command succeeds",
                    test_type_fn: |s| TestType::CodeExecution(s.to_string()),
                },
                CruxTemplate {
                    pattern: "lookup",
                    test_type_fn: |s| TestType::Lookup(s.to_string()),
                },
            ],
        }
    }

    /// Extract a crux for a given target claim
    pub fn extract(&self, graph: &ConstraintGraph, target: ClaimId) -> Option<Crux> {
        let claim = graph.get(target)?;

        if !claim.alive || claim.tested {
            return None;
        }

        // Get conflicting claims
        let conflicts = graph.get_conflicts(target);

        // Generate falsifier based on claim content
        let (falsifier, test_type) = self.generate_falsifier(&claim.content);

        // Determine outcomes
        let if_true = if conflicts.is_empty() {
            format!("Accept '{}'", claim.content)
        } else {
            format!(
                "Accept '{}', kill {} alternatives",
                claim.content,
                conflicts.len()
            )
        };

        let if_false = format!("Kill '{}', explore alternatives", claim.content);

        Some(Crux {
            target,
            falsifier,
            test_type,
            if_true,
            if_false,
            expected: if claim.probability > 0.5 {
                Some(true)
            } else {
                Some(false)
            },
        })
    }

    /// Generate a falsifier and test type from claim content
    fn generate_falsifier(&self, content: &str) -> (String, TestType) {
        let lower = content.to_lowercase();

        // Pattern matching for common claim types
        if lower.contains("file") || lower.contains("exists") {
            let path = self.extract_path(content).unwrap_or("*".to_string());
            return (
                format!("Check if '{}' exists", path),
                TestType::FileCheck(path),
            );
        }

        if lower.contains("compiles") || lower.contains("builds") {
            return (
                format!("Attempt to compile: {}", content),
                TestType::CodeExecution("cargo check 2>&1".to_string()),
            );
        }

        if lower.contains("test") || lower.contains("passes") {
            return (
                format!("Run tests for: {}", content),
                TestType::CodeExecution("cargo test 2>&1".to_string()),
            );
        }

        if lower.contains("=") || lower.contains("equals") {
            return (
                format!("Verify equality: {}", content),
                TestType::Lookup(content.to_string()),
            );
        }

        // Default: manual verification
        (format!("Manually verify: {}", content), TestType::Manual)
    }

    fn extract_path(&self, content: &str) -> Option<String> {
        // Simple heuristic: look for quoted strings or paths
        if let Some(start) = content.find('\'') {
            if let Some(end) = content[start + 1..].find('\'') {
                return Some(content[start + 1..start + 1 + end].to_string());
            }
        }
        if let Some(start) = content.find('"') {
            if let Some(end) = content[start + 1..].find('"') {
                return Some(content[start + 1..start + 1 + end].to_string());
            }
        }
        // Look for file-like patterns
        for word in content.split_whitespace() {
            if word.contains('/') || word.contains('\\') || word.contains('.') {
                return Some(word.to_string());
            }
        }
        None
    }

    /// Create a crux manually
    pub fn manual(
        target: ClaimId,
        falsifier: &str,
        test_type: TestType,
        if_true: &str,
        if_false: &str,
    ) -> Crux {
        Crux {
            target,
            falsifier: falsifier.to_string(),
            test_type,
            if_true: if_true.to_string(),
            if_false: if_false.to_string(),
            expected: None,
        }
    }
}

impl Default for CruxExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_file_crux() {
        let mut graph = ConstraintGraph::new();
        let c = graph.add_claim("file 'Cargo.toml' exists", 0.9, 1.0);

        let extractor = CruxExtractor::new();
        let crux = extractor.extract(&graph, c).unwrap();

        assert!(matches!(crux.test_type, TestType::FileCheck(_)));
    }

    #[test]
    fn test_compile_crux() {
        let mut graph = ConstraintGraph::new();
        let c = graph.add_claim("code compiles successfully", 0.7, 5.0);

        let extractor = CruxExtractor::new();
        let crux = extractor.extract(&graph, c).unwrap();

        assert!(matches!(crux.test_type, TestType::CodeExecution(_)));
    }
}
