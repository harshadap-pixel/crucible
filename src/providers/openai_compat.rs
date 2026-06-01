/// OpenAI-compatible provider.
///
/// Works with any endpoint that implements `/v1/chat/completions` with SSE streaming:
///   - OpenAI (api.openai.com)
///   - Groq (api.groq.com/openai/v1)
///   - Together AI (api.together.xyz/v1)
///   - Mistral (api.mistral.ai/v1)
///   - OpenRouter (openrouter.ai/api/v1)
///   - LM Studio, vLLM, llama.cpp server, Ollama compat mode
use anyhow::{bail, Result};
use async_trait::async_trait;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

use super::{CompletionResult, Provider};

// ── Request types ─────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<Message>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    stream_options: StreamOptions,
}

#[derive(Serialize)]
struct StreamOptions {
    include_usage: bool,
}

#[derive(Serialize, Clone)]
struct Message {
    role: String,
    content: String,
}

// ── Response types (streaming chunks) ────────────────────────────────────────

#[derive(Deserialize)]
struct ChatChunk {
    #[serde(default)]
    choices: Vec<ChunkChoice>,
    #[serde(default)]
    usage: Option<Usage>,
}

#[derive(Deserialize)]
struct ChunkChoice {
    delta: Delta,
}

#[derive(Deserialize, Default)]
struct Delta {
    #[serde(default)]
    content: Option<String>,
}

#[derive(Deserialize, Default)]
struct Usage {
    #[serde(default)]
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u32,
}

// ── Client ────────────────────────────────────────────────────────────────────

pub struct OpenAICompatProvider {
    base_url: String,
    api_key: String,
    client: reqwest::Client,
}

impl OpenAICompatProvider {
    pub fn new(base_url: &str, api_key: &str) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key: api_key.to_string(),
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(300))
                .build()
                .unwrap(),
        }
    }

    async fn do_chat(
        &self,
        model: &str,
        messages: Vec<Message>,
        temperature: f32,
    ) -> Result<CompletionResult> {
        let body = ChatRequest {
            model,
            messages,
            stream: true,
            temperature: Some(temperature),
            max_tokens: Some(2048),
            stream_options: StreamOptions {
                include_usage: true,
            },
        };

        let t0 = Instant::now();
        let resp = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?;

        if !resp.status().is_success() {
            bail!(
                "OpenAI-compat error {}: {}",
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

                // SSE lines start with "data: "
                let Some(payload) = line.strip_prefix("data:") else {
                    continue;
                };
                let payload = payload.trim();
                if payload == "[DONE]" {
                    break;
                }
                if payload.is_empty() {
                    continue;
                }

                if let Ok(chunk) = serde_json::from_str::<ChatChunk>(payload) {
                    // Accumulate content
                    for choice in &chunk.choices {
                        if let Some(ref content) = choice.delta.content {
                            if !content.is_empty() {
                                if !ttft_recorded {
                                    ttft_ms = t0.elapsed().as_millis() as u64;
                                    ttft_recorded = true;
                                }
                                text.push_str(content);
                            }
                        }
                    }
                    // Usage is in the last chunk (when include_usage=true)
                    if let Some(usage) = chunk.usage {
                        input_tokens = usage.prompt_tokens;
                        output_tokens = usage.completion_tokens;
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
impl Provider for OpenAICompatProvider {
    async fn chat(
        &self,
        model: &str,
        system: Option<&str>,
        user: &str,
        temperature: f32,
    ) -> Result<CompletionResult> {
        let mut messages = Vec::new();
        if let Some(sys) = system {
            messages.push(Message {
                role: "system".into(),
                content: sys.into(),
            });
        }
        messages.push(Message {
            role: "user".into(),
            content: user.into(),
        });
        self.do_chat(model, messages, temperature).await
    }

    async fn chat_with_history(
        &self,
        model: &str,
        history: &[(String, String)],
        final_user: &str,
        temperature: f32,
    ) -> Result<CompletionResult> {
        let mut messages: Vec<Message> = history
            .iter()
            .map(|(role, content)| Message {
                role: role.clone(),
                content: content.clone(),
            })
            .collect();
        messages.push(Message {
            role: "user".into(),
            content: final_user.into(),
        });
        self.do_chat(model, messages, temperature).await
    }

    fn name(&self) -> &'static str {
        "openai-compat"
    }
}
