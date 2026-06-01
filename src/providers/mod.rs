pub mod anthropic;
pub mod ollama;
pub mod openai_compat;
pub mod script;
#[cfg(test)]
mod tests;

pub use ollama::{LocalModel, OllamaClient};

use anyhow::Result;
use async_trait::async_trait;
use std::sync::Arc;

// ── Shared completion result ──────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct CompletionResult {
    pub text: String,
    pub latency_ms: u128,
    /// Time from request sent to first token received (0 if not applicable).
    pub ttft_ms: u64,
    pub input_tokens: u32,
    pub output_tokens: u32,
}

// ── Provider trait ────────────────────────────────────────────────────────────

/// Any completion backend must implement this trait.
#[async_trait]
pub trait Provider: Send + Sync {
    /// Single-turn chat completion.
    async fn chat(
        &self,
        model: &str,
        system: Option<&str>,
        user: &str,
        temperature: f32,
    ) -> Result<CompletionResult>;

    /// Multi-turn chat completion.
    /// Default implementation concatenates history into a single user message;
    /// providers that support native multi-turn should override this.
    async fn chat_with_history(
        &self,
        model: &str,
        history: &[(String, String)],
        final_user: &str,
        temperature: f32,
    ) -> Result<CompletionResult> {
        let mut combined = String::new();
        for (role, content) in history {
            combined.push_str(&format!("{}: {}\n", role, content));
        }
        combined.push_str(&format!("user: {}", final_user));
        self.chat(model, None, &combined, temperature).await
    }

    /// Short identifier shown in output (e.g. "ollama", "openai", "anthropic").
    fn name(&self) -> &'static str;
}

// ── Model reference (provider + resolved model name) ─────────────────────────

/// A resolved model: the provider to call and the bare model name to pass to it.
///
/// Example:
///   `ModelRef::resolve("openai:gpt-4o", "http://localhost:11434")`
///   → `{ provider: OpenAICompatProvider, model: "gpt-4o" }`
pub struct ModelRef {
    pub provider: Arc<dyn Provider>,
    pub model: String,
}

impl ModelRef {
    /// Parse a model spec string and return a ModelRef.
    ///
    /// Supported prefixes:
    ///   (none)           → Ollama   (backward-compatible default)
    ///   ollama:          → Ollama   (explicit)
    ///   openai:          → OpenAI   (OPENAI_API_KEY)
    ///   groq:            → Groq     (GROQ_API_KEY)
    ///   together:        → Together (TOGETHER_API_KEY)
    ///   mistral:         → Mistral  (MISTRAL_API_KEY)
    ///   openrouter:      → OpenRouter (OPENROUTER_API_KEY)
    ///   anthropic:       → Anthropic (ANTHROPIC_API_KEY)
    pub fn resolve(spec: &str, ollama_url: &str) -> Result<Self> {
        // Known cloud prefixes — check these before trying to split on ':'
        // because Ollama model tags also use ':' (e.g. "llama3.1:8b").
        let cloud_prefixes: &[(&str, &str, &str, &str)] = &[
            (
                "openai:",
                "https://api.openai.com/v1",
                "OPENAI_API_KEY",
                "openai",
            ),
            (
                "groq:",
                "https://api.groq.com/openai/v1",
                "GROQ_API_KEY",
                "groq",
            ),
            (
                "together:",
                "https://api.together.xyz/v1",
                "TOGETHER_API_KEY",
                "together",
            ),
            (
                "mistral:",
                "https://api.mistral.ai/v1",
                "MISTRAL_API_KEY",
                "mistral",
            ),
            (
                "openrouter:",
                "https://openrouter.ai/api/v1",
                "OPENROUTER_API_KEY",
                "openrouter",
            ),
        ];

        for (prefix, base_url, key_env, _pname) in cloud_prefixes {
            if let Some(model) = spec.strip_prefix(prefix) {
                let api_key = std::env::var(key_env).unwrap_or_else(|_| {
                    eprintln!(
                        "  ⚠ {} not set — {} calls may fail",
                        key_env,
                        prefix.trim_end_matches(':')
                    );
                    String::new()
                });
                let provider =
                    Arc::new(openai_compat::OpenAICompatProvider::new(base_url, &api_key));
                return Ok(Self {
                    provider,
                    model: model.to_string(),
                });
            }
        }

        if let Some(model) = spec.strip_prefix("anthropic:") {
            let api_key = std::env::var("ANTHROPIC_API_KEY").unwrap_or_else(|_| {
                eprintln!("  ⚠ ANTHROPIC_API_KEY not set — Anthropic calls may fail");
                String::new()
            });
            let provider = Arc::new(anthropic::AnthropicProvider::new(&api_key));
            return Ok(Self {
                provider,
                model: model.to_string(),
            });
        }

        // Azure OpenAI — deployment name after prefix, e.g. "azure:gpt-4o"
        // Reads AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION.
        if let Some(deployment) = spec.strip_prefix("azure:") {
            let api_key = std::env::var("AZURE_OPENAI_API_KEY").unwrap_or_else(|_| {
                eprintln!("  ⚠ AZURE_OPENAI_API_KEY not set — Azure calls may fail");
                String::new()
            });
            let endpoint = std::env::var("AZURE_OPENAI_ENDPOINT").unwrap_or_else(|_| {
                eprintln!("  ⚠ AZURE_OPENAI_ENDPOINT not set — Azure calls may fail");
                String::new()
            });
            let api_version = std::env::var("AZURE_OPENAI_API_VERSION")
                .unwrap_or_else(|_| "2025-01-01-preview".to_string());
            // Azure URL: {endpoint}/openai/deployments/{deployment}
            let base_url = format!(
                "{}/openai/deployments/{}",
                endpoint.trim_end_matches('/'),
                deployment
            );
            let provider = Arc::new(openai_compat::OpenAICompatProvider::new_azure(
                &base_url,
                &api_key,
                &api_version,
                deployment,
            ));
            return Ok(Self {
                provider,
                model: deployment.to_string(),
            });
        }

        // "ollama:" explicit prefix or bare model name (backward-compat default)
        let model = spec.strip_prefix("ollama:").unwrap_or(spec).to_string();
        let provider = Arc::new(OllamaClient::new(ollama_url));
        Ok(Self { provider, model })
    }
}

// ── Provider detection ────────────────────────────────────────────────────────

/// One detected provider — either local Ollama or a cloud provider with a key set.
#[derive(Debug, Clone)]
pub struct AvailableProvider {
    /// Human-readable name: "ollama", "openai", "groq", etc.
    pub name: &'static str,
    /// Model spec strings ready to use (e.g. "gpt-4o-mini", "groq:llama-3.1-8b-instant")
    pub models: Vec<String>,
    /// Whether an API key / local server is confirmed present
    pub configured: bool,
}

/// Cloud provider registry: (prefix, env-key, display-name, cheap-default-model)
static CLOUD_PROVIDERS: &[(&str, &str, &str, &str)] = &[
    ("groq", "GROQ_API_KEY", "groq", "groq:llama-3.1-8b-instant"),
    ("openai", "OPENAI_API_KEY", "openai", "openai:gpt-4o-mini"),
    (
        "anthropic",
        "ANTHROPIC_API_KEY",
        "anthropic",
        "anthropic:claude-3-5-haiku-latest",
    ),
    (
        "mistral",
        "MISTRAL_API_KEY",
        "mistral",
        "mistral:mistral-small-latest",
    ),
    (
        "together",
        "TOGETHER_API_KEY",
        "together",
        "together:meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    ),
    (
        "openrouter",
        "OPENROUTER_API_KEY",
        "openrouter",
        "openrouter:openai/gpt-4o-mini",
    ),
];

/// Azure OpenAI is detected separately because it needs two env vars
/// (key + endpoint) and the default model depends on what's deployed.
const AZURE_KEY_ENV: &str = "AZURE_OPENAI_API_KEY";
const AZURE_ENDPOINT_ENV: &str = "AZURE_OPENAI_ENDPOINT";

/// Scan the local environment and return every provider that is usable right now.
///
/// - Pings Ollama and lists its models.
/// - Checks each cloud provider's env var.
pub async fn detect_available(ollama_url: &str) -> Vec<AvailableProvider> {
    let mut out: Vec<AvailableProvider> = Vec::new();

    // ── Ollama ────────────────────────────────────────────────────────────────
    let client = OllamaClient::new(ollama_url);
    let local_models = client.list_models().await;
    let ollama_ok = !local_models.is_empty();
    out.push(AvailableProvider {
        name: "ollama",
        models: local_models.iter().map(|m| m.name.clone()).collect(),
        configured: ollama_ok,
    });

    // ── Cloud providers ───────────────────────────────────────────────────────
    for (_, key_env, name, default_model) in CLOUD_PROVIDERS {
        let key_set = std::env::var(key_env)
            .map(|v| !v.is_empty())
            .unwrap_or(false);
        out.push(AvailableProvider {
            name,
            models: if key_set {
                vec![default_model.to_string()]
            } else {
                vec![]
            },
            configured: key_set,
        });
    }

    // ── Azure OpenAI ──────────────────────────────────────────────────────────
    // Needs both key AND endpoint to be configured.
    let azure_key = std::env::var(AZURE_KEY_ENV)
        .map(|v| !v.is_empty())
        .unwrap_or(false);
    let azure_endpoint = std::env::var(AZURE_ENDPOINT_ENV)
        .map(|v| !v.is_empty())
        .unwrap_or(false);
    let azure_ok = azure_key && azure_endpoint;
    out.push(AvailableProvider {
        name: "azure",
        models: if azure_ok {
            // We don't know which deployments exist without calling the management API,
            // so show a placeholder — user specifies the deployment name explicitly.
            vec!["azure:gpt-4o".to_string()]
        } else {
            vec![]
        },
        configured: azure_ok,
    });

    out
}

/// Coerce the judge model when it is still the default Ollama model but the eval
/// model is a cloud provider.  This prevents failures when Ollama is not running
/// but a cloud API key is set.
///
/// Rule: if `judge_spec` equals the default (`llama3.1:8b`) AND `eval_spec` has
/// a recognised cloud prefix, mirror the eval provider with its cheapest model.
pub fn coerce_judge(judge_spec: &str, eval_spec: &str) -> String {
    const DEFAULT_JUDGE: &str = "llama3.1:8b";
    if judge_spec != DEFAULT_JUDGE {
        return judge_spec.to_string();
    }
    let cloud_mirror: &[(&str, &str)] = &[
        ("openai:", "openai:gpt-4o-mini"),
        ("groq:", "groq:llama-3.1-8b-instant"),
        ("anthropic:", "anthropic:claude-haiku-4-5-20251001"),
        ("mistral:", "mistral:mistral-small-latest"),
        (
            "together:",
            "together:meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        ),
        ("openrouter:", "openrouter:openai/gpt-4o-mini"),
        // Azure: mirror with the cheapest commonly-deployed model.
        // User can always override with --judge azure:{deployment}.
        ("azure:", "azure:gpt-4o-mini"),
    ];
    for (prefix, cheap) in cloud_mirror {
        if eval_spec.starts_with(prefix) {
            return cheap.to_string();
        }
    }
    judge_spec.to_string()
}

/// Pick the best available model spec string.
///
/// Priority: Ollama (local, free) → Groq (fast free tier) → OpenAI → Anthropic → …
/// Returns `None` if nothing is configured.
pub fn auto_select_model(providers: &[AvailableProvider]) -> Option<String> {
    // Ollama first — pick the first available local model
    if let Some(ollama) = providers.iter().find(|p| p.name == "ollama") {
        if let Some(model) = ollama.models.first() {
            return Some(model.clone());
        }
    }
    // Cloud: walk in priority order (same order as CLOUD_PROVIDERS)
    for (_, _, name, _) in CLOUD_PROVIDERS {
        if let Some(p) = providers.iter().find(|p| p.name == *name && p.configured) {
            if let Some(model) = p.models.first() {
                return Some(model.clone());
            }
        }
    }
    None
}
