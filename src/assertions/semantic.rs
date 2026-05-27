use crate::assertions::AssertionResult;
use crate::providers::ollama::cosine_similarity;
use crate::providers::OllamaClient;
use anyhow::Result;

const EMBED_MODEL: &str = "nomic-embed-text";

pub async fn check(
    client: &OllamaClient,
    output: &str,
    reference: &str,
    threshold: f64,
    weight: f64,
) -> Result<AssertionResult> {
    let emb_out = client
        .embed(EMBED_MODEL, output)
        .await
        .unwrap_or_else(|_| vec![]);
    let emb_ref = client
        .embed(EMBED_MODEL, reference)
        .await
        .unwrap_or_else(|_| vec![]);

    if emb_out.is_empty() || emb_ref.is_empty() {
        return Ok(AssertionResult {
            kind: "semantic".into(),
            passed: false,
            score: 0.0,
            reason: format!("Embedding failed — is '{}' pulled in Ollama?", EMBED_MODEL),
            weight,
        });
    }

    let sim = cosine_similarity(&emb_out, &emb_ref) as f64;
    let passed = sim >= threshold;

    Ok(AssertionResult {
        kind: "semantic".into(),
        passed,
        score: sim.clamp(0.0, 1.0),
        reason: format!("Cosine similarity {:.3} (threshold {:.2})", sim, threshold),
        weight,
    })
}
