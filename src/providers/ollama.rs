use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

// ── Request / Response types ──────────────────────────────────────────────────

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<Message<'a>>,
    stream: bool,
    options: ChatOptions,
}

#[derive(Serialize)]
struct ChatOptions {
    temperature: f32,
    num_predict: i32,
}

#[derive(Serialize, Deserialize)]
struct Message<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Deserialize)]
struct ChatResponse {
    message: MessageContent,
    #[serde(default)]
    prompt_eval_count: u32,
    #[serde(default)]
    eval_count: u32,
}

#[derive(Deserialize)]
struct MessageContent {
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

// ── Completion result ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct CompletionResult {
    pub text: String,
    pub latency_ms: u128,
    pub input_tokens: u32,
    pub output_tokens: u32,
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
                .timeout(Duration::from_secs(120))
                .build()
                .unwrap(),
        }
    }

    /// Single-turn chat completion
    pub async fn chat(
        &self,
        model: &str,
        system: Option<&str>,
        user: &str,
        temperature: f32,
    ) -> Result<CompletionResult> {
        let mut messages = Vec::new();
        if let Some(sys) = system {
            messages.push(Message {
                role: "system",
                content: sys,
            });
        }
        messages.push(Message {
            role: "user",
            content: user,
        });

        let body = ChatRequest {
            model,
            messages,
            stream: false,
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

        let data: ChatResponse = resp.json().await?;
        Ok(CompletionResult {
            text: data.message.content,
            latency_ms: t0.elapsed().as_millis(),
            input_tokens: data.prompt_eval_count,
            output_tokens: data.eval_count,
        })
    }

    /// Multi-turn chat — replays a full conversation history then sends final turn.
    /// `history` is a list of (role, content) pairs in order.
    /// The final user message should NOT be included in history; pass it as `final_user`.
    pub async fn chat_with_history(
        &self,
        model: &str,
        history: &[(String, String)],
        final_user: &str,
        temperature: f32,
    ) -> Result<CompletionResult> {
        let mut messages: Vec<Message> = history
            .iter()
            .map(|(role, content)| Message { role, content })
            .collect();
        messages.push(Message {
            role: "user",
            content: final_user,
        });

        let body = ChatRequest {
            model,
            messages,
            stream: false,
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

        let data: ChatResponse = resp.json().await?;
        Ok(CompletionResult {
            text: data.message.content,
            latency_ms: t0.elapsed().as_millis(),
            input_tokens: data.prompt_eval_count,
            output_tokens: data.eval_count,
        })
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
        // ~400-token shared prefix
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
            likely_active: ratio < 0.60, // warm ≥40% faster = cache likely
        })
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
