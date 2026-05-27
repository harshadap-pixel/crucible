use crate::assertions::AssertionResult;
use crate::providers::OllamaClient;
use anyhow::Result;

const SYSTEM: &str = "\
You are a strict evaluator. You will be given an AI response and a rubric. \
Score the response from 0.0 to 1.0 based on how well it satisfies the rubric. \
Reply with ONLY a JSON object: {\"score\": <float>, \"reason\": \"<one sentence>\"}. \
No other text.";

pub async fn check(
    client: &OllamaClient,
    judge: &str,
    output: &str,
    rubric: &str,
    threshold: f64,
    weight: f64,
) -> Result<AssertionResult> {
    let prompt =
        format!("RUBRIC:\n{rubric}\n\nRESPONSE TO EVALUATE:\n{output}\n\nScore the response.");

    // Retry up to 3 times — smaller models sometimes return malformed JSON
    let mut last_err = String::new();
    for attempt in 0..3u8 {
        match client.chat(judge, Some(SYSTEM), &prompt, 0.0).await {
            Ok(result) => {
                if let Some((score, reason)) = parse_judge_response(&result.text) {
                    let passed = score >= threshold;
                    return Ok(AssertionResult {
                        kind: "llm_judge".into(),
                        passed,
                        score,
                        reason,
                        weight,
                    });
                }
                last_err = format!("Bad JSON on attempt {attempt}: {}", result.text);
            }
            Err(e) => {
                last_err = e.to_string();
            }
        }
    }

    // Fallback: judge failed to respond properly
    Ok(AssertionResult {
        kind: "llm_judge".into(),
        passed: false,
        score: 0.0,
        reason: format!("Judge returned unparseable output: {last_err}"),
        weight,
    })
}

fn parse_judge_response(text: &str) -> Option<(f64, String)> {
    // Look for JSON object anywhere in the response
    let start = text.find('{')?;
    let end = text.rfind('}')? + 1;
    let json_str = &text[start..end];

    let v: serde_json::Value = serde_json::from_str(json_str).ok()?;
    let score = v.get("score")?.as_f64()?;
    let reason = v.get("reason")?.as_str()?.to_string();
    Some((score.clamp(0.0, 1.0), reason))
}
