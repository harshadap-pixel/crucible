//! Tree-Sitter query patterns for detecting AI code constructs
//! Queries match specific AST patterns for eval runners, RAG pipelines, NL2SQL, etc.

/// Queries for TypeScript/JavaScript eval runners
pub mod typescript {
    pub const EVAL_RUNNER_PATTERNS: &[(&str, &str)] = &[
        // Pattern: test() or it() or describe() blocks (Jest/Mocha)
        (
            "test_blocks",
            r#"(call_expression
                function: (identifier) @func_name
                (#match? @func_name "^(test|it|describe)$")
            )"#,
        ),
        // Pattern: assert or expect calls
        (
            "assertions",
            r#"(call_expression
                function: (member_expression
                    object: (identifier) @obj
                    property: (property_identifier) @method
                )
                (#match? @method "^(assert|expect|toBe|toEqual)$")
            )"#,
        ),
        // Pattern: process.exit() calls (run completion marker)
        (
            "process_exit",
            r#"(call_expression
                function: (member_expression
                    object: (identifier) @obj
                    property: (property_identifier) @method
                )
                (#eq? @obj "process")
                (#eq? @method "exit")
            )"#,
        ),
    ];

    pub const RAG_PATTERNS: &[(&str, &str)] = &[
        // Pattern: vector store initialization (Pinecone, Weaviate, etc)
        (
            "vector_stores",
            r#"(new_expression
                constructor: (identifier) @class
                (#match? @class "^(Pinecone|Weaviate|Milvus|QdrantClient|SupabaseVectorStore)$")
            )"#,
        ),
        // Pattern: embedding calls
        (
            "embeddings",
            r#"(call_expression
                function: (member_expression
                    property: (property_identifier) @method
                )
                (#match? @method "^(embed|getEmbedding|embedQuery)$")
            )"#,
        ),
        // Pattern: similarity search
        (
            "similarity_search",
            r#"(call_expression
                function: (member_expression
                    property: (property_identifier) @method
                )
                (#match? @method "^(search|similaritySearch|query)$")
            )"#,
        ),
    ];
}

/// Queries for Python eval runners
pub mod python {
    pub const EVAL_RUNNER_PATTERNS: &[(&str, &str)] = &[
        // Pattern: pytest test functions
        (
            "pytest_functions",
            r#"(function_definition
                name: (identifier) @func
                (#match? @func "^test_")
            )"#,
        ),
        // Pattern: unittest TestCase classes
        (
            "unittest_classes",
            r#"(class_definition
                name: (identifier) @class
                bases: (argument_list
                    (identifier) @base
                )
                (#match? @base "TestCase")
            )"#,
        ),
        // Pattern: assert statements
        ("assertions", r#"(assert_statement)"#),
        // Pattern: sys.exit() calls
        (
            "sys_exit",
            r#"(call
                function: (attribute
                    object: (identifier) @obj
                    attr: (identifier) @method
                )
                (#eq? @obj "sys")
                (#eq? @method "exit")
            )"#,
        ),
    ];

    pub const RAG_PATTERNS: &[(&str, &str)] = &[
        // Pattern: vector store imports/usage
        (
            "vector_stores",
            r#"(import_statement
                module: (dotted_name) @module
                (#match? @module ".*vector.*")
            )"#,
        ),
        // Pattern: embedding function calls
        (
            "embeddings",
            r#"(call
                function: (attribute
                    attr: (identifier) @method
                )
                (#match? @method "^(embed|get_embedding)$")
            )"#,
        ),
        // Pattern: similarity search
        (
            "similarity_search",
            r#"(call
                function: (attribute
                    attr: (identifier) @method
                )
                (#match? @method "^(search|similarity_search)$")
            )"#,
        ),
    ];
}

/// Queries for NL2SQL validators
pub mod nl2sql {
    pub const VALIDATOR_PATTERNS: &[(&str, &str)] = &[
        // TypeScript: validateSQL, checkSQL function
        (
            "validate_function",
            r#"(function_declaration
                name: (identifier) @func
                (#match? @func ".*[Vv]alidate.*[Ss]ql|[Cc]heck.*[Ss]ql.*")
            )"#,
        ),
        // Python: validate_sql function
        (
            "validate_function_py",
            r#"(function_definition
                name: (identifier) @func
                (#match? @func ".*[Vv]alidate.*[Ss]ql|[Cc]heck.*[Ss]ql.*")
            )"#,
        ),
    ];
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_typescript_queries_exist() {
        assert!(!typescript::EVAL_RUNNER_PATTERNS.is_empty());
        assert!(!typescript::RAG_PATTERNS.is_empty());
    }

    #[test]
    fn test_python_queries_exist() {
        assert!(!python::EVAL_RUNNER_PATTERNS.is_empty());
        assert!(!python::RAG_PATTERNS.is_empty());
    }

    #[test]
    fn test_nl2sql_queries_exist() {
        assert!(!nl2sql::VALIDATOR_PATTERNS.is_empty());
    }

    #[test]
    fn test_query_format() {
        // Verify all queries are non-empty strings
        for (_, query) in typescript::EVAL_RUNNER_PATTERNS {
            assert!(!query.is_empty(), "Query should not be empty");
        }
    }
}
