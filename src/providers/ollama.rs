use anyhow::{bail, Result};
use async_trait::async_trait;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

use super::{CompletionResult, Provider};

// ── Request / Response types ──────────────────────────────────────────────────

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<OwnedMessage>,
    stream: bool,
    options: ChatOptions,
}

#[derive(Serialize)]
struct ChatOptions {
    temperature: f32,
    num_predict: i32,
}

#[derive(Serialize, Clone)]
struct OwnedMessage {
    role: String,
    content: String,
}

/// One chunk from Ollama's streaming NDJSON response.
#[derive(Deserialize)]
struct StreamChunk {
    message: StreamMessageContent,
    #[serde(default)]
    done: bool,
    /// Only present on the final chunk (done=true)
    #[serde(default)]
    prompt_eval_count: u32,
    #[serde(default)]
    eval_count: u32,
}

#[derive(Deserialize)]
struct StreamMessageContent {
    content: String,
}

#[derive(Serialize)]
struct EmbedRequest<'a> {
    model: &'a str,
    input: &'a str,
}

#[derive(Deserialize)]
struct EmbedResponse {
    embeddings: Vec<Vec<f32>>,
}

// ── Model list ────────────────────────────────────────────────────────────────

#[derive(Deserialize, Debug, Clone)]
pub struct LocalModel {
    pub name: String,
    /// Size in bytes
    #[serde(default)]
    pub size: u64,
    #[serde(default)]
    pub details: LocalModelDetails,
}

#[derive(Deserialize, Debug, Clone, Default)]
pub struct LocalModelDetails {
    #[serde(default)]
    pub parameter_size: String,
    #[serde(default)]
    pub quantization_level: String,
}

#[derive(Deserialize)]
struct TagsResponse {
    #[serde(default)]
    models: Vec<LocalModel>,
}

// ── Model show / metadata ─────────────────────────────────────────────────────

#[derive(Deserialize, Debug, Clone, Default)]
pub struct ModelInfo {
    #[serde(default)]
    pub modelinfo: serde_json::Map<String, serde_json::Value>,
    #[serde(default)]
    pub details: ModelDetails,
}

#[derive(Deserialize, Debug, Clone, Default)]
pub struct ModelDetails {
    #[serde(default)]
    pub parameter_size: String,
    #[serde(default)]
    pub quantization_level: String,
    #[allow(dead_code)]
    #[serde(default)]
    pub families: Vec<String>,
}

// ── Client ────────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct OllamaClient {
    base_url: String,
    client: reqwest::Client,
}

impl OllamaClient {
    pub fn new(base_url: &str) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(300))
                .build()
                .unwrap(),
        }
    }

    // ── Internal streaming helper ─────────────────────────────────────────────

    /// Send a chat request using Ollama's streaming API and measure TTFT.
    ///
    /// Returns the full concatenated text plus timing / token counts.
    async fn send_streaming(
        &self,
        model: &str,
        messages: Vec<OwnedMessage>,
        temperature: f32,
    ) -> Result<CompletionResult> {
        let body = ChatRequest {
            model: model.to_string(),
            messages,
            stream: true,
            options: ChatOptions {
                temperature,
                num_predict: 2048,
            },
        };

        let t0 = Instant::now();
        let resp = self
            .client
            .post(format!("{}/api/chat", self.base_url))
            .json(&body)
            .send()
            .await?;

        if !resp.status().is_success() {
            bail!(
                "Ollama chat error {}: {}",
                resp.status(),
                resp.text().await?
            );
        }

        let mut byte_stream = resp.bytes_stream();
        let mut text = String::new();
        let mut ttft_ms: u64 = 0;
        let mut ttft_recorded = false;
        let mut input_tokens = 0u32;
        let mut output_tokens = 0u32;
        // Buffer for partial lines across chunks
        let mut line_buf = String::new();

        while let Some(chunk) = byte_stream.next().await {
            let bytes = chunk?;
            let s = std::str::from_utf8(&bytes).unwrap_or("");
            line_buf.push_str(s);

            // Process every complete '\n'-terminated line
            while let Some(nl) = line_buf.find('\n') {
                let line = line_buf[..nl].trim().to_string();
                line_buf.drain(..=nl);

                if line.is_empty() {
                    continue;
                }

                if let Ok(chunk) = serde_json::from_str::<StreamChunk>(&line) {
                    // Record TTFT on the first non-empty content token
                    if !ttft_recorded && !chunk.message.content.is_empty() {
                        ttft_ms = t0.elapsed().as_millis() as u64;
                        ttft_recorded = true;
                    }
                    text.push_str(&chunk.message.content);

                    if chunk.done {
                        input_tokens = chunk.prompt_eval_count;
                        output_tokens = chunk.eval_count;
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

    // ── Public API ────────────────────────────────────────────────────────────

    /// Single-turn chat completion — measures TTFT via streaming.
    pub async fn chat(
        &self,
        model: &str,
        system: Option<&str>,
        user: &str,
        temperature: f32,
    ) -> Result<CompletionResult> {
        let mut messages = Vec::new();
        if let Some(sys) = system {
            messages.push(OwnedMessage {
                role: "system".into(),
                content: sys.into(),
            });
        }
        messages.push(OwnedMessage {
            role: "user".into(),
            content: user.into(),
        });
        self.send_streaming(model, messages, temperature).await
    }

    /// Multi-turn chat — measures TTFT via streaming.
    pub async fn chat_with_history(
        &self,
        model: &str,
        history: &[(String, String)],
        final_user: &str,
        temperature: f32,
    ) -> Result<CompletionResult> {
        let mut messages: Vec<OwnedMessage> = history
            .iter()
            .map(|(role, content)| OwnedMessage {
                role: role.clone(),
                content: content.clone(),
            })
            .collect();
        messages.push(OwnedMessage {
            role: "user".into(),
            content: final_user.into(),
        });
        self.send_streaming(model, messages, temperature).await
    }

    /// Compute embeddings for a single string
    pub async fn embed(&self, model: &str, text: &str) -> Result<Vec<f32>> {
        let body = EmbedRequest { model, input: text };
        let resp = self
            .client
            .post(format!("{}/api/embed", self.base_url))
            .json(&body)
            .send()
            .await?;

        if !resp.status().is_success() {
            bail!(
                "Ollama embed error {}: {}",
                resp.status(),
                resp.text().await?
            );
        }

        let data: EmbedResponse = resp.json().await?;
        data.embeddings
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("Empty embedding response"))
    }

    /// Fetch model metadata from /api/show
    pub async fn show(&self, model: &str) -> Result<ModelInfo> {
        let resp = self
            .client
            .post(format!("{}/api/show", self.base_url))
            .json(&serde_json::json!({ "model": model }))
            .send()
            .await?;

        if !resp.status().is_success() {
            bail!(
                "Ollama show error {}: {}",
                resp.status(),
                resp.text().await?
            );
        }

        Ok(resp.json::<ModelInfo>().await?)
    }

    /// Probe KV cache by timing two calls with the same long prefix
    pub async fn probe_kv_cache(&self, model: &str) -> Result<KvCacheProbeResult> {
        let prefix = "The field of artificial intelligence has a long history that began in the \
            1950s when Alan Turing proposed his famous test for machine intelligence. Since then, \
            the field has gone through several waves of optimism and funding cuts, often called \
            AI winters, before the deep learning revolution of the 2010s brought renewed interest. \
            Modern large language models are trained on vast corpora of text and learn to predict \
            the next token in a sequence. This capability, combined with instruction fine-tuning \
            and reinforcement learning from human feedback, produces models that can follow \
            complex instructions and engage in nuanced reasoning across a wide variety of tasks. \
            The compute required to train these models has grown exponentially, with the largest \
            models now requiring thousands of specialized accelerators running for months. ";

        let cold = self
            .chat(
                model,
                None,
                &format!("{prefix} Question A: What is machine learning?"),
                0.0,
            )
            .await?;

        let warm = self
            .chat(
                model,
                None,
                &format!("{prefix} Question B: What is deep learning?"),
                0.0,
            )
            .await?;

        let ratio = warm.latency_ms as f64 / cold.latency_ms.max(1) as f64;

        Ok(KvCacheProbeResult {
            cold_latency_ms: cold.latency_ms,
            warm_latency_ms: warm.latency_ms,
            speedup_ratio: 1.0 / ratio,
            likely_active: ratio < 0.60,
        })
    }

    /// List all locally available models from /api/tags.
    /// Returns an empty vec if Ollama is unreachable.
    pub async fn list_models(&self) -> Vec<LocalModel> {
        let Ok(resp) = self
            .client
            .get(format!("{}/api/tags", self.base_url))
            .send()
            .await
        else {
            return vec![];
        };
        if !resp.status().is_success() {
            return vec![];
        }
        resp.json::<TagsResponse>()
            .await
            .map(|r| r.models)
            .unwrap_or_default()
    }

    /// Check if Ollama is reachable
    pub async fn health(&self) -> bool {
        self.client
            .get(format!("{}/api/tags", self.base_url))
            .send()
            .await
            .map(|r| r.status().is_success())
            .unwrap_or(false)
    }
}

#[derive(Debug)]
pub struct KvCacheProbeResult {
    pub cold_latency_ms: u128,
    pub warm_latency_ms: u128,
    pub speedup_ratio: f64,
    pub likely_active: bool,
}

// ── Cosine similarity helper ──────────────────────────────────────────────────

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if mag_a == 0.0 || mag_b == 0.0 {
        0.0
    } else {
        dot / (mag_a * mag_b)
    }
}

// ── Provider trait impl ───────────────────────────────────────────────────────

#[async_trait]
impl Provider for OllamaClient {
    async fn chat(
        &self,
        model: &str,
        system: Option<&str>,
        user: &str,
        temperature: f32,
    ) -> Result<CompletionResult> {
        self.chat(model, system, user, temperature).await
    }

    async fn chat_with_history(
        &self,
        model: &str,
        history: &[(String, String)],
        final_user: &str,
        temperature: f32,
    ) -> Result<CompletionResult> {
        self.chat_with_history(model, history, final_user, temperature)
            .await
    }

    fn name(&self) -> &'static str {
        "ollama"
    }
}
