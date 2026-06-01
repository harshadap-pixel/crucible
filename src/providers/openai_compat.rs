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
    /// Legacy parameter — used by OpenAI-compat providers (Groq, Mistral, Together, etc.)
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    /// Newer parameter required by GPT-4.1, o-series, and Azure AI Foundry models.
    /// Mutually exclusive with max_tokens — only one is sent per request.
    #[serde(skip_serializing_if = "Option::is_none")]
    max_completion_tokens: Option<u32>,
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
    /// Azure OpenAI requires `api-key` header; all other providers use `Authorization: Bearer`.
    use_api_key_header: bool,
    /// Azure OpenAI requires `?api-version=` on every request.
    api_version: Option<String>,
    /// GPT-4.1 / o-series / Azure AI Foundry models require `max_completion_tokens`
    /// instead of the legacy `max_tokens` parameter.
    use_max_completion_tokens: bool,
    /// o-series / reasoning models reject any explicit temperature value — omit it entirely.
    omit_temperature: bool,
    /// Human-readable name returned by `Provider::name()`.
    provider_name: &'static str,
    client: reqwest::Client,
}

impl OpenAICompatProvider {
    pub fn new(base_url: &str, api_key: &str) -> Self {
        Self::new_full(
            base_url,
            api_key,
            false,
            None,
            false,
            false,
            "openai-compat",
        )
    }

    /// Full constructor used for Azure OpenAI.
    ///
    /// `base_url` for Azure should be:
    ///   `https://{resource}.openai.azure.com/openai/deployments/{deployment}`
    ///
    /// The provider appends `/chat/completions?api-version={api_version}` automatically.
    /// Uses `max_completion_tokens` (required by GPT-4.1 and newer Azure-hosted models).
    /// Omits `temperature` for o-series / reasoning models that reject explicit values.
    pub fn new_azure(base_url: &str, api_key: &str, api_version: &str, deployment: &str) -> Self {
        // o-series / reasoning model deployment names contain "o1", "o3", "codex",
        // or start with "gpt-5" — these reject explicit temperature values.
        let omit_temp = is_reasoning_deployment(deployment);
        Self::new_full(
            base_url,
            api_key,
            true,
            Some(api_version.to_string()),
            true, // Azure models require max_completion_tokens
            omit_temp,
            "azure",
        )
    }

    fn new_full(
        base_url: &str,
        api_key: &str,
        use_api_key_header: bool,
        api_version: Option<String>,
        use_max_completion_tokens: bool,
        omit_temperature: bool,
        provider_name: &'static str,
    ) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key: api_key.to_string(),
            use_api_key_header,
            api_version,
            use_max_completion_tokens,
            omit_temperature,
            provider_name,
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
            // o-series / reasoning models reject any explicit temperature — omit entirely.
            temperature: if self.omit_temperature {
                None
            } else {
                Some(temperature)
            },
            max_tokens: if self.use_max_completion_tokens {
                None
            } else {
                Some(2048)
            },
            max_completion_tokens: if self.use_max_completion_tokens {
                Some(2048)
            } else {
                None
            },
            stream_options: StreamOptions {
                include_usage: true,
            },
        };

        // Build URL — Azure appends ?api-version=, others don't.
        let url = match &self.api_version {
            Some(ver) => format!("{}/chat/completions?api-version={}", self.base_url, ver),
            None => format!("{}/chat/completions", self.base_url),
        };

        let t0 = Instant::now();
        let mut req = self
            .client
            .post(&url)
            .header("Content-Type", "application/json");

        // Auth header: Azure uses `api-key`, everything else uses `Authorization: Bearer`.
        req = if self.use_api_key_header {
            req.header("api-key", &self.api_key)
        } else {
            req.header("Authorization", format!("Bearer {}", self.api_key))
        };

        let resp = req.json(&body).send().await?;

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
        self.provider_name
    }
}

/// Heuristic: does this Azure deployment name belong to an o-series / reasoning
/// model that rejects explicit `temperature` values?
///
/// Matches: o1*, o3*, *codex*, gpt-5*, *-chat (reasoning variants)
pub fn is_reasoning_deployment(deployment: &str) -> bool {
    let d = deployment.to_lowercase();
    d.starts_with("o1")
        || d.starts_with("o3")
        || d.contains("codex")
        || d.starts_with("gpt-5")
        || d.ends_with("-chat")
}
