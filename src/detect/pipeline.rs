use anyhow::Result;
use std::path::Path;

pub struct PipelineScan {
    pub signals: Vec<(String, bool)>,
}

/// Scan a pipeline config file (any text format) for known mechanism signals
pub fn scan(path: &str) -> Result<PipelineScan> {
    if !Path::new(path).exists() {
        anyhow::bail!("Pipeline config not found: {path}");
    }

    let content = std::fs::read_to_string(path)?.to_lowercase();

    // (Display name, signal keywords — all matched case-insensitively)
    let checks: &[(&str, &[&str])] = &[
        // ── Vector / retrieval ──────────────────────────────────────────────
        (
            "HyDE",
            &["hyde", "hypothetical_document", "generate_hypothetical"],
        ),
        (
            "HNSW / USearch",
            &[
                "hnswlib",
                "ef_construction",
                "hnsw",
                "usearch",
                "metrickind.cos",
                "metrickind::cos",
                "ef_construction",
                "connectivity:",
            ],
        ),
        (
            "Chunking (fixed)",
            &["chunk_size", "chunk_length", "fixed_size", "chunk_overlap"],
        ),
        (
            "Chunking (recursive)",
            &[
                "recursivecharactertextsplitter",
                "recursive_splitter",
                "recursive_chunk",
            ],
        ),
        (
            "Chunking (semantic)",
            &["semanticchunker", "semantic_chunk", "sentence_split"],
        ),
        (
            "Chunking (sliding)",
            &["sliding_window", "stride_chunk", "window_size"],
        ),
        (
            "Reranking",
            &["rerank", "cross_encoder", "reranker", "cohere.rerank"],
        ),
        (
            "Query expansion",
            &[
                "query_expansion",
                "multi_query",
                "n_queries",
                "expand_query",
            ],
        ),
        (
            "Semantic normalizer",
            &[
                "query_normalize",
                "spell_correct",
                "synonym_expand",
                "normalizesql",
                "normalize_query",
            ],
        ),
        // ── Embeddings ──────────────────────────────────────────────────────
        (
            "ONNX embeddings",
            &[
                "onnxruntime",
                "ort.inferencesession",
                "onnxruntime-web",
                "ort.tensor",
                "inferencesession.create",
                ".onnx",
            ],
        ),
        (
            "Sentence embeddings",
            &[
                "all-minilm",
                "bge-",
                "text-embedding-",
                "embed_model",
                "createembedding",
                "embed(text",
                "embedbatch",
            ],
        ),
        (
            "Mean pooling",
            &[
                "meanpool",
                "mean_pool",
                "pooler_output",
                "last_hidden_state",
            ],
        ),
        // ── LLM providers ───────────────────────────────────────────────────
        (
            "AWS Bedrock agent",
            &[
                "bedrock-agent-runtime",
                "invokeagentcommand",
                "bedrockagentruntime",
                "bedrock-runtime",
                "createbedrockagentclient",
            ],
        ),
        (
            "OpenAI / Azure OAI",
            &[
                "openai",
                "azureopenai",
                "chatcompletion",
                "createchatcompletion",
                "gpt-4",
                "gpt-3.5",
            ],
        ),
        (
            "Anthropic Claude",
            &["anthropic", "claude-3", "claude-sonnet", "claude-opus"],
        ),
        // ── Safety / guardrails ─────────────────────────────────────────────
        (
            "SQL safety guardrail",
            &[
                "validatandsanitizesql",
                "validateandsanitizesql",
                "allowed_schemas",
                "dangerous_keywords",
                "allowedschemas",
                "dangerousKeywords",
            ],
        ),
        (
            "LlamaGuard",
            &["llama_guard", "llamaguard", "meta-llama/llama-guard"],
        ),
        (
            "NeMo Guardrails",
            &["nemo_guardrails", "nemoguardrails", "colang"],
        ),
        (
            "PII redaction",
            &["presidio", "piianaly", "anonymize", "redact_pii"],
        ),
        // ── MCP (Model Context Protocol) ────────────────────────────────────
        // Framework
        (
            "MCP server (@rekog/mcp-nest)",
            &[
                "@rekog/mcp-nest",
                "rekog/mcp-nest",
                "mcpmodule.forroot",
                "mcpmodule",
            ],
        ),
        (
            "MCP server (raw SDK)",
            &[
                "@modelcontextprotocol/sdk",
                "modelcontextprotocol/sdk",
                "listtools requestschema",
                "calltoolrequestschema",
            ],
        ),
        (
            "MCP server (FastMCP/Python)",
            &[
                "fastmcp",
                "@mcp.tool",
                "mcp.server.fastmcp",
                "from mcp import",
            ],
        ),
        // Transports
        (
            "MCP transport: SSE",
            &[
                "sseservertransport",
                "mcptransporttype.sse",
                "sseendpoint",
                "mcp-sse",
                "mcp_sse_service",
                "text/event-stream",
            ],
        ),
        (
            "MCP transport: Streamable HTTP",
            &[
                "streamablehttpservertransport",
                "mcptransporttype.streamable_http",
                "streamablehttp",
                "streamable-http",
                "mcp-streamable-http",
            ],
        ),
        (
            "MCP transport: Stdio",
            &[
                "stdioservertransport",
                "stdioclienttransport",
                "mcptransporttype.stdio",
                "stdio.service",
            ],
        ),
        // Tool authoring style
        (
            "MCP @Tool decorator",
            &[
                "@tool({",
                "@tool( {",
                "from '@rekog/mcp-nest'",
                "from \"@rekog/mcp-nest\"",
            ],
        ),
        (
            "MCP server.tool() API",
            &["server.tool(", "server.resource(", "server.prompt("],
        ),
        // ── Infrastructure ──────────────────────────────────────────────────
        (
            "RLHF fine-tuned",
            &["rlhf", "reinforcement", "reward_model", "ppo"],
        ),
        (
            "Prompt caching",
            &[
                "prompt_cache",
                "cache_prompt",
                "cache_prefix",
                "cachepoint",
                "cache_point",
                "cachecontrol",
                "cache_system_prompt",
            ],
        ),
        (
            "RAG pipeline",
            &[
                "retrieval",
                "vectorsearch",
                "vector_search",
                "semantic_search",
                "nearest_neighbor",
                "top_k",
            ],
        ),
        (
            "Multi-turn memory",
            &[
                "conversationhistory",
                "conversation_history",
                "thread_context",
                "compactthreadcontext",
                "messagehistory",
                "chat_history",
            ],
        ),
        (
            "Tool / function call",
            &[
                "tool_call",
                "function_call",
                "tool-definitions",
                "tool-types",
                "available_tools",
                "calltool",
            ],
        ),
    ];

    let signals = checks
        .iter()
        .map(|(name, keywords)| {
            let found = keywords.iter().any(|kw| content.contains(kw));
            (name.to_string(), found)
        })
        .collect();

    Ok(PipelineScan { signals })
}
