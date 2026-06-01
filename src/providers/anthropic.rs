/// Anthropic Messages API provider.
///
/// Uses Anthropic's native streaming API (`/v1/messages`) with SSE.
/// API key is read from the `ANTHROPIC_API_KEY` environment variable.
///
/// Note: Anthropic does not expose an embeddings endpoint.
/// Semantic assertions will fall back to Ollama for embeddings regardless
/// of whether the eval model is Anthropic.
use anyhow::{bail, Result};
use async_trait::async_trait;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

use super::{CompletionResult, Provider};

const ANTHROPIC_VERSION: &str = "2023-06-01";
const BASE_URL: &str = "https://api.anthropic.com";

// ── Request types ─────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct MessagesRequest<'a> {
    model: &'a str,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<&'a str>,
    max_tokens: u32,
    stream: bool,
}

#[derive(Serialize, Clone)]
struct AnthropicMessage {
    role: String,
    content: String,
}

// ── SSE event types ───────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct SseEvent {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    delta: Option<ContentDelta>,
    #[serde(default)]
    usage: Option<AnthropicUsage>,
    // message_start contains the input usage
    #[serde(default)]
    message: Option<MessageStart>,
}

/// Shape of the `delta` object varies by event type:
///   content_block_delta → `{"type":"text_delta","text":"..."}`
///   message_delta       → `{"stop_reason":"end_turn"}` (no `type` field)
/// We default `kind` to empty string so the latter shape deserialises without error.
#[derive(Deserialize, Default)]
struct ContentDelta {
    #[serde(rename = "type", default)]
    kind: String,
    #[serde(default)]
    text: Option<String>,
}

#[derive(Deserialize)]
struct AnthropicUsage {
    #[serde(default)]
    output_tokens: u32,
}

#[derive(Deserialize)]
struct MessageStart {
    #[serde(default)]
    usage: Option<MessageUsage>,
}

#[derive(Deserialize)]
struct MessageUsage {
    #[serde(default)]
    input_tokens: u32,
}

// ── Client ────────────────────────────────────────────────────────────────────

pub struct AnthropicProvider {
    api_key: String,
    /// Base URL (without trailing slash). Defaults to `https://api.anthropic.com`.
    /// Overridable via `new_with_base_url` — used in unit tests to point at a mock server.
    base_url: String,
    client: reqwest::Client,
}

impl AnthropicProvider {
    pub fn new(api_key: &str) -> Self {
        Self::new_with_base_url(api_key, BASE_URL)
    }

    /// Construct with a custom base URL — intended for unit tests that spin up a
    /// mock HTTP server.
    pub fn new_with_base_url(api_key: &str, base_url: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: base_url.trim_end_matches('/').to_string(),
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(300))
                .build()
                .unwrap(),
        }
    }

    async fn do_chat(
        &self,
        model: &str,
        system: Option<&str>,
        messages: Vec<AnthropicMessage>,
    ) -> Result<CompletionResult> {
        let body = MessagesRequest {
            model,
            messages,
            system,
            max_tokens: 2048,
            stream: true,
        };

        let t0 = Instant::now();
        let resp = self
            .client
            .post(format!("{}/v1/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?;

        if !resp.status().is_success() {
            bail!(
                "Anthropic API error {}: {}",
                resp.status(),
                resp.text().await?
            );
        }

        let mut stream = resp.bytes_stream();
        let mut text = String::new();
        let mut ttft_ms: u64 = 0;
        let mut ttft_recorded = false;
        let mut input_tokens = 0u32;
        let mut output_tokens = 0u32;
        let mut line_buf = String::new();

        while let Some(chunk) = stream.next().await {
            let bytes = chunk?;
            let s = std::str::from_utf8(&bytes).unwrap_or("");
            line_buf.push_str(s);

            while let Some(nl) = line_buf.find('\n') {
                let line = line_buf[..nl].trim().to_string();
                line_buf.drain(..=nl);

                // Anthropic SSE: "data: {...}"
                let Some(payload) = line.strip_prefix("data:") else {
                    continue;
                };
                let payload = payload.trim();
                if payload.is_empty() {
                    continue;
                }

                if let Ok(ev) = serde_json::from_str::<SseEvent>(payload) {
                    match ev.kind.as_str() {
                        "message_start" => {
                            if let Some(msg) = ev.message {
                                if let Some(usage) = msg.usage {
                                    input_tokens = usage.input_tokens;
                                }
                            }
                        }
                        "content_block_delta" => {
                            if let Some(delta) = ev.delta {
                                if delta.kind == "text_delta" {
                                    if let Some(t) = delta.text {
                                        if !t.is_empty() {
                                            if !ttft_recorded {
                                                ttft_ms = t0.elapsed().as_millis() as u64;
                                                ttft_recorded = true;
                                            }
                                            text.push_str(&t);
                                        }
                                    }
                                }
                            }
                        }
                        "message_delta" => {
                            if let Some(usage) = ev.usage {
                                output_tokens = usage.output_tokens;
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        Ok(CompletionResult {
            text,
            latency_ms: t0.elapsed().as_millis(),
            ttft_ms: if ttft_recorded {
                ttft_ms
            } else {
                t0.elapsed().as_millis() as u64
            },
            input_tokens,
            output_tokens,
        })
    }
}

#[async_trait]
impl Provider for AnthropicProvider {
    async fn chat(
        &self,
        model: &str,
        system: Option<&str>,
        user: &str,
        _temperature: f32,
    ) -> Result<CompletionResult> {
        // Anthropic doesn't take temperature in the same field;
        // temperature can be added to the body if needed via an extension.
        let messages = vec![AnthropicMessage {
            role: "user".into(),
            content: user.into(),
        }];
        self.do_chat(model, system, messages).await
    }

    async fn chat_with_history(
        &self,
        model: &str,
        history: &[(String, String)],
        final_user: &str,
        _temperature: f32,
    ) -> Result<CompletionResult> {
        let mut messages: Vec<AnthropicMessage> = history
            .iter()
            .map(|(role, content)| AnthropicMessage {
                role: role.clone(),
                content: content.clone(),
            })
            .collect();
        messages.push(AnthropicMessage {
            role: "user".into(),
            content: final_user.into(),
        });
        self.do_chat(model, None, messages).await
    }

    fn name(&self) -> &'static str {
        "anthropic"
    }
}
