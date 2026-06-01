/// Provider unit tests — use `httpmock` to serve fake SSE responses so these
/// tests run without real API keys or a live Ollama instance.
#[cfg(test)]
mod provider_tests {
    use httpmock::prelude::*;

    use crate::providers::{
        anthropic::AnthropicProvider, coerce_judge, openai_compat::OpenAICompatProvider, ModelRef,
        Provider,
    };

    // ── ModelRef::resolve ─────────────────────────────────────────────────────

    #[test]
    fn resolve_bare_name_is_ollama() {
        let mr = ModelRef::resolve("llama3.1:8b", "http://localhost:11434").unwrap();
        assert_eq!(mr.provider.name(), "ollama");
        assert_eq!(mr.model, "llama3.1:8b");
    }

    #[test]
    fn resolve_explicit_ollama_prefix() {
        let mr = ModelRef::resolve("ollama:mistral", "http://localhost:11434").unwrap();
        assert_eq!(mr.provider.name(), "ollama");
        assert_eq!(mr.model, "mistral");
    }

    #[test]
    fn resolve_openai_prefix() {
        let mr = ModelRef::resolve("openai:gpt-4o-mini", "http://localhost:11434").unwrap();
        assert_eq!(mr.provider.name(), "openai-compat");
        assert_eq!(mr.model, "gpt-4o-mini");
    }

    #[test]
    fn resolve_groq_prefix() {
        let mr = ModelRef::resolve("groq:llama-3.1-8b-instant", "http://localhost:11434").unwrap();
        assert_eq!(mr.provider.name(), "openai-compat");
        assert_eq!(mr.model, "llama-3.1-8b-instant");
    }

    #[test]
    fn resolve_anthropic_prefix() {
        let mr = ModelRef::resolve(
            "anthropic:claude-3-5-haiku-latest",
            "http://localhost:11434",
        )
        .unwrap();
        assert_eq!(mr.provider.name(), "anthropic");
        assert_eq!(mr.model, "claude-3-5-haiku-latest");
    }

    // ── coerce_judge ──────────────────────────────────────────────────────────

    #[test]
    fn coerce_judge_leaves_custom_judge_alone() {
        let result = coerce_judge("openai:gpt-4o-mini", "anthropic:claude-3-5-haiku-latest");
        assert_eq!(result, "openai:gpt-4o-mini");
    }

    #[test]
    fn coerce_judge_mirrors_openai_eval() {
        let result = coerce_judge("llama3.1:8b", "openai:gpt-4o");
        assert_eq!(result, "openai:gpt-4o-mini");
    }

    #[test]
    fn coerce_judge_mirrors_anthropic_eval() {
        let result = coerce_judge("llama3.1:8b", "anthropic:claude-sonnet-4-6");
        assert_eq!(result, "anthropic:claude-haiku-4-5-20251001");
    }

    #[test]
    fn coerce_judge_leaves_ollama_eval_alone() {
        let result = coerce_judge("llama3.1:8b", "llama3.2:3b");
        assert_eq!(result, "llama3.1:8b");
    }

    #[test]
    fn coerce_judge_mirrors_groq_eval() {
        let result = coerce_judge("llama3.1:8b", "groq:llama-3.1-70b-versatile");
        assert_eq!(result, "groq:llama-3.1-8b-instant");
    }

    // ── OpenAICompatProvider::chat — mocked SSE ───────────────────────────────

    #[tokio::test]
    async fn openai_compat_chat_parses_sse_response() {
        let server = MockServer::start_async().await;

        // Minimal valid SSE streaming response
        let sse_body = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}],\"usage\":null}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\", world!\"}}],\"usage\":null}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5}}\n\n",
            "data: [DONE]\n\n",
        );

        let _mock = server.mock(|when, then| {
            when.method(POST).path("/chat/completions");
            then.status(200)
                .header("Content-Type", "text/event-stream")
                .body(sse_body);
        });

        let provider = OpenAICompatProvider::new(&server.base_url(), "test-key");
        let result = provider
            .chat("gpt-4o-mini", None, "Say hello", 0.0)
            .await
            .unwrap();

        assert_eq!(result.text, "Hello, world!");
        assert_eq!(result.input_tokens, 10);
        assert_eq!(result.output_tokens, 5);
    }

    #[tokio::test]
    async fn openai_compat_chat_sends_system_message() {
        let server = MockServer::start_async().await;

        let sse_body =
            "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}],\"usage\":null}\n\ndata: [DONE]\n\n";

        let _mock = server.mock(|when, then| {
            when.method(POST)
                .path("/chat/completions")
                .body_contains("\"role\":\"system\"");
            then.status(200)
                .header("Content-Type", "text/event-stream")
                .body(sse_body);
        });

        let provider = OpenAICompatProvider::new(&server.base_url(), "test-key");
        let result = provider
            .chat(
                "gpt-4o-mini",
                Some("You are a helpful assistant."),
                "Hi",
                0.0,
            )
            .await
            .unwrap();

        assert!(!result.text.is_empty());
    }

    #[tokio::test]
    async fn openai_compat_returns_error_on_non_200() {
        let server = MockServer::start_async().await;

        let _mock = server.mock(|when, then| {
            when.method(POST).path("/chat/completions");
            then.status(401).body("{\"error\":\"Unauthorized\"}");
        });

        let provider = OpenAICompatProvider::new(&server.base_url(), "bad-key");
        let err = provider
            .chat("gpt-4o-mini", None, "Hello", 0.0)
            .await
            .unwrap_err();

        assert!(err.to_string().contains("401"));
    }

    // ── AnthropicProvider::chat — mocked SSE ─────────────────────────────────

    #[tokio::test]
    async fn anthropic_chat_parses_sse_response() {
        let server = MockServer::start_async().await;

        // Anthropic native SSE format
        let sse_body = concat!(
            "event: message_start\n",
            "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":8,\"output_tokens\":0}}}\n\n",
            "event: content_block_start\n",
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi there\"}}\n\n",
            "event: message_delta\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":3}}\n\n",
            "event: message_stop\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        );

        let _mock = server.mock(|when, then| {
            when.method(POST).path("/v1/messages");
            then.status(200)
                .header("Content-Type", "text/event-stream")
                .body(sse_body);
        });

        let provider = AnthropicProvider::new_with_base_url("test-key", &server.base_url());
        let result = provider
            .chat("claude-3-5-haiku-latest", None, "Say hi", 0.0)
            .await
            .unwrap();

        assert_eq!(result.text, "Hi there");
        assert_eq!(result.input_tokens, 8);
        assert_eq!(result.output_tokens, 3);
    }

    #[tokio::test]
    async fn anthropic_chat_returns_error_on_non_200() {
        let server = MockServer::start_async().await;

        let _mock = server.mock(|when, then| {
            when.method(POST).path("/v1/messages");
            then.status(401)
                .body("{\"type\":\"error\",\"error\":{\"type\":\"authentication_error\"}}");
        });

        let provider = AnthropicProvider::new_with_base_url("bad-key", &server.base_url());
        let err = provider
            .chat("claude-3-5-haiku-latest", None, "Hi", 0.0)
            .await
            .unwrap_err();

        assert!(err.to_string().contains("401"));
    }
}
