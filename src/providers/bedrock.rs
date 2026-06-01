/// AWS Bedrock provider — uses the Bedrock Converse API with SigV4 request signing.
///
/// The Converse API is a unified interface that works across all Bedrock model
/// families (Anthropic Claude, Meta Llama, Mistral, Amazon Titan, Cohere, etc.)
/// without needing model-specific request formats.
///
/// Endpoint: POST https://bedrock-runtime.{region}.amazonaws.com/model/{model-id}/converse
///
/// Auth: AWS SigV4 — computed from AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY.
/// Optionally uses AWS_SESSION_TOKEN for temporary credentials (IAM roles, SSO).
///
/// Model IDs use the full Bedrock format, e.g.:
///   anthropic.claude-3-5-sonnet-20241022-v2:0
///   meta.llama3-70b-instruct-v1:0
///   mistral.mistral-large-2402-v1:0
///   amazon.titan-text-express-v1
use anyhow::{bail, Result};
use async_trait::async_trait;
use hmac::{Hmac, Mac};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use super::{CompletionResult, Provider};

type HmacSha256 = Hmac<Sha256>;

const SERVICE: &str = "bedrock";
const ALGORITHM: &str = "AWS4-HMAC-SHA256";

// ── Request types (Converse API) ──────────────────────────────────────────────

#[derive(Serialize)]
struct ConverseRequest {
    messages: Vec<ConverseMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<Vec<SystemBlock>>,
    #[serde(rename = "inferenceConfig")]
    inference_config: InferenceConfig,
}

#[derive(Serialize, Deserialize, Clone)]
struct ConverseMessage {
    role: String,
    content: Vec<ContentBlock>,
}

#[derive(Serialize, Deserialize, Clone)]
struct ContentBlock {
    text: String,
}

#[derive(Serialize)]
struct SystemBlock {
    text: String,
}

#[derive(Serialize)]
struct InferenceConfig {
    #[serde(rename = "maxTokens")]
    max_tokens: u32,
}

// ── Response types ────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct ConverseResponse {
    output: ConverseOutput,
    usage: ConverseUsage,
}

#[derive(Deserialize)]
struct ConverseOutput {
    message: ConverseMessage,
}

#[derive(Deserialize)]
struct ConverseUsage {
    #[serde(rename = "inputTokens", default)]
    input_tokens: u32,
    #[serde(rename = "outputTokens", default)]
    output_tokens: u32,
}

// ── SigV4 helpers ─────────────────────────────────────────────────────────────

fn sha256_hex(data: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(data);
    hex_encode(&h.finalize())
}

fn hmac_sha256(key: &[u8], data: &[u8]) -> Vec<u8> {
    let mut mac = HmacSha256::new_from_slice(key).expect("HMAC accepts any key length");
    mac.update(data);
    mac.finalize().into_bytes().to_vec()
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Derive the SigV4 signing key.
/// Key = HMAC(HMAC(HMAC(HMAC("AWS4" + secret, date), region), service), "aws4_request")
fn signing_key(secret: &str, date: &str, region: &str, service: &str) -> Vec<u8> {
    let k_date = hmac_sha256(format!("AWS4{secret}").as_bytes(), date.as_bytes());
    let k_region = hmac_sha256(&k_date, region.as_bytes());
    let k_service = hmac_sha256(&k_region, service.as_bytes());
    hmac_sha256(&k_service, b"aws4_request")
}

/// Build the AWS4-HMAC-SHA256 Authorization header for a Bedrock request.
fn sigv4_auth(
    access_key: &str,
    secret_key: &str,
    session_token: Option<&str>,
    region: &str,
    method: &str,
    path: &str, // URL path, e.g. /model/anthropic.claude.../converse
    payload: &[u8],
    datetime: &str, // e.g. "20240101T120000Z"
    date: &str,     // e.g. "20240101"
) -> (String, String) {
    let host = format!("bedrock-runtime.{region}.amazonaws.com");
    let payload_hash = sha256_hex(payload);

    // ── Signed headers (must be sorted) ──────────────────────────────────────
    let mut signed_header_names = vec!["content-type", "host", "x-amz-date"];
    if session_token.is_some() {
        signed_header_names.push("x-amz-security-token");
    }
    signed_header_names.sort_unstable();
    let signed_headers = signed_header_names.join(";");

    // ── Canonical headers (name:value\n, sorted) ──────────────────────────────
    let mut header_pairs: Vec<(String, String)> = vec![
        ("content-type".into(), "application/json".into()),
        ("host".into(), host.clone()),
        ("x-amz-date".into(), datetime.into()),
    ];
    if let Some(tok) = session_token {
        header_pairs.push(("x-amz-security-token".into(), tok.into()));
    }
    header_pairs.sort_by(|a, b| a.0.cmp(&b.0));
    let canonical_headers: String = header_pairs
        .iter()
        .map(|(k, v)| format!("{k}:{v}\n"))
        .collect();

    // ── Canonical request ─────────────────────────────────────────────────────
    let canonical_request =
        format!("{method}\n{path}\n\n{canonical_headers}\n{signed_headers}\n{payload_hash}");

    // ── Credential scope ──────────────────────────────────────────────────────
    let credential_scope = format!("{date}/{region}/{SERVICE}/aws4_request");

    // ── String to sign ────────────────────────────────────────────────────────
    let string_to_sign = format!(
        "{ALGORITHM}\n{datetime}\n{credential_scope}\n{}",
        sha256_hex(canonical_request.as_bytes())
    );

    // ── Signature ─────────────────────────────────────────────────────────────
    let key = signing_key(secret_key, date, region, SERVICE);
    let signature = hex_encode(&hmac_sha256(&key, string_to_sign.as_bytes()));

    let auth = format!(
        "{ALGORITHM} Credential={access_key}/{credential_scope}, \
         SignedHeaders={signed_headers}, Signature={signature}"
    );

    (auth, host)
}

// ── Provider ──────────────────────────────────────────────────────────────────

pub struct BedrockProvider {
    access_key: String,
    secret_key: String,
    /// Optional session token for temporary credentials (IAM roles / AWS SSO).
    session_token: Option<String>,
    region: String,
    client: reqwest::Client,
}

impl BedrockProvider {
    pub fn new(
        access_key: &str,
        secret_key: &str,
        session_token: Option<String>,
        region: &str,
    ) -> Self {
        Self {
            access_key: access_key.to_string(),
            secret_key: secret_key.to_string(),
            session_token,
            region: region.to_string(),
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(300))
                .build()
                .unwrap(),
        }
    }

    async fn do_converse(
        &self,
        model_id: &str,
        messages: Vec<ConverseMessage>,
        system: Option<&str>,
    ) -> Result<CompletionResult> {
        let body = ConverseRequest {
            messages,
            system: system.map(|s| {
                vec![SystemBlock {
                    text: s.to_string(),
                }]
            }),
            inference_config: InferenceConfig { max_tokens: 2048 },
        };

        let payload = serde_json::to_vec(&body)?;
        let path = format!("/model/{model_id}/converse");
        let url = format!(
            "https://bedrock-runtime.{}.amazonaws.com{}",
            self.region, path
        );

        // Datetime strings for SigV4
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let datetime = format_datetime(now);
        let date = &datetime[..8];

        let (auth_header, host) = sigv4_auth(
            &self.access_key,
            &self.secret_key,
            self.session_token.as_deref(),
            &self.region,
            "POST",
            &path,
            &payload,
            &datetime,
            date,
        );

        let t0 = std::time::Instant::now();
        let mut req = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .header("Host", &host)
            .header("X-Amz-Date", &datetime)
            .header("Authorization", &auth_header);

        if let Some(ref tok) = self.session_token {
            req = req.header("X-Amz-Security-Token", tok);
        }

        let resp = req.body(payload).send().await?;
        let latency_ms = t0.elapsed().as_millis();

        if !resp.status().is_success() {
            bail!("Bedrock error {}: {}", resp.status(), resp.text().await?);
        }

        let parsed: ConverseResponse = resp.json().await?;

        // Extract text from the first content block
        let text = parsed
            .output
            .message
            .content
            .into_iter()
            .map(|b| b.text)
            .collect::<Vec<_>>()
            .join("");

        Ok(CompletionResult {
            text,
            latency_ms,
            // Bedrock Converse API is non-streaming — TTFT equals total latency.
            ttft_ms: latency_ms as u64,
            input_tokens: parsed.usage.input_tokens,
            output_tokens: parsed.usage.output_tokens,
        })
    }
}

#[async_trait]
impl Provider for BedrockProvider {
    async fn chat(
        &self,
        model: &str,
        system: Option<&str>,
        user: &str,
        _temperature: f32,
    ) -> Result<CompletionResult> {
        let messages = vec![ConverseMessage {
            role: "user".into(),
            content: vec![ContentBlock { text: user.into() }],
        }];
        self.do_converse(model, messages, system).await
    }

    async fn chat_with_history(
        &self,
        model: &str,
        history: &[(String, String)],
        final_user: &str,
        _temperature: f32,
    ) -> Result<CompletionResult> {
        let mut messages: Vec<ConverseMessage> = history
            .iter()
            .map(|(role, content)| ConverseMessage {
                role: role.clone(),
                content: vec![ContentBlock {
                    text: content.clone(),
                }],
            })
            .collect();
        messages.push(ConverseMessage {
            role: "user".into(),
            content: vec![ContentBlock {
                text: final_user.into(),
            }],
        });
        self.do_converse(model, messages, None).await
    }

    fn name(&self) -> &'static str {
        "bedrock"
    }
}

// ── Datetime formatting ───────────────────────────────────────────────────────

/// Format a Unix timestamp as SigV4 datetime string: "20240101T120000Z"
fn format_datetime(secs: u64) -> String {
    // Manual UTC formatting — avoids pulling in chrono for a single format.
    let s = secs;
    let sec = s % 60;
    let min = (s / 60) % 60;
    let hour = (s / 3600) % 24;
    let days = s / 86400;

    // Compute year/month/day from days since epoch (1970-01-01)
    let (year, month, day) = days_to_ymd(days);

    format!("{year:04}{month:02}{day:02}T{hour:02}{min:02}{sec:02}Z")
}

/// Convert days since Unix epoch (1970-01-01) to (year, month, day).
///
/// Uses Howard Hinnant's civil_from_days algorithm.  That algorithm counts
/// days from the proleptic Gregorian epoch 0000-03-01; we shift the input by
/// the Unix-epoch offset (719468 days) before applying it.
fn days_to_ymd(days: u64) -> (u64, u64, u64) {
    // Shift Unix days → Gregorian calendar days (epoch = 0000-03-01).
    // 1970-01-01 is day 719468 in the proleptic Gregorian calendar.
    let mut z = days + 719468;

    // 400-year Gregorian cycle = 146097 days
    let era = z / 146097;
    z %= 146097;
    let yoe = (z - z / 1460 + z / 36524 - z / 146096) / 365;
    let y = yoe + era * 400;
    let doy = z - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn datetime_format_epoch() {
        // Unix epoch = 1970-01-01T00:00:00Z
        assert_eq!(format_datetime(0), "19700101T000000Z");
    }

    #[test]
    fn datetime_format_known_date() {
        // 2024-01-15 11:30:45 UTC = 1705318245
        assert_eq!(format_datetime(1705318245), "20240115T113045Z");
    }

    #[test]
    fn signing_key_deterministic() {
        // Same inputs always produce same key
        let k1 = signing_key(
            "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
            "20150830",
            "us-east-1",
            "iam",
        );
        let k2 = signing_key(
            "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
            "20150830",
            "us-east-1",
            "iam",
        );
        assert_eq!(k1, k2);
    }

    #[test]
    fn hex_encode_correct() {
        assert_eq!(hex_encode(&[0xde, 0xad, 0xbe, 0xef]), "deadbeef");
    }
}
