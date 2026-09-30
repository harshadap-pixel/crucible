use crate::assertions::AssertionResult;
use crate::providers::ModelRef;
use anyhow::Result;

const SYSTEM: &str = "\
You are a strict evaluator. You will be given an AI response and a rubric. \
Score the response from 0.0 to 1.0 based on how well it satisfies the rubric. \
Reply with ONLY a JSON object: {\"score\": <float>, \"reason\": \"<one sentence>\"}. \
No other text.";

pub async fn check(
    judge: &ModelRef,
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
        match judge
            .provider
            .chat(&judge.model, Some(SYSTEM), &prompt, 0.0)
            .await
        {
            Ok(result) => {
                if let Some((score, reason)) = parse_judge_response(&result.text) {
                    let passed = score >= threshold;
                    return Ok(AssertionResult {
                        kind: "llm_judge".into(),
                        passed,
                        score,
                        reason,
                        weight,
                        errored: false,
                    });
                }
                last_err = format!("Bad JSON on attempt {attempt}: {}", result.text);
            }
            Err(e) => {
                last_err = e.to_string();
            }
        }
    }

    // Judge never produced a usable score: report the assertion as errored so a
    // broken judge isn't mistaken for a bad model answer.
    Ok(AssertionResult {
        kind: "llm_judge".into(),
        passed: false,
        score: 0.0,
        reason: format!("Judge error after 3 attempts: {last_err}"),
        weight,
        errored: true,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::{CompletionResult, Provider};
    use async_trait::async_trait;
    use std::sync::Arc;

    /// Judge stub that always replies with the same text.
    struct FixedJudge(&'static str);

    #[async_trait]
    impl Provider for FixedJudge {
        async fn chat(
            &self,
            _model: &str,
            _system: Option<&str>,
            _user: &str,
            _temperature: f32,
        ) -> Result<CompletionResult> {
            Ok(CompletionResult {
                text: self.0.into(),
                latency_ms: 0,
                ttft_ms: 0,
                input_tokens: 0,
                output_tokens: 0,
            })
        }

        fn name(&self) -> &'static str {
            "fixed"
        }
    }

    fn judge(reply: &'static str) -> ModelRef {
        ModelRef {
            provider: Arc::new(FixedJudge(reply)),
            model: "stub".into(),
        }
    }

    #[tokio::test]
    async fn broken_judge_is_errored_not_just_failed() {
        let r = check(&judge("not json at all"), "out", "rubric", 0.5, 1.0)
            .await
            .unwrap();
        assert!(!r.passed);
        assert!(r.errored);
    }

    #[tokio::test]
    async fn low_score_is_a_failure_not_an_error() {
        let r = check(
            &judge(r#"{"score": 0.1, "reason": "bad"}"#),
            "out",
            "rubric",
            0.5,
            1.0,
        )
        .await
        .unwrap();
        assert!(!r.passed);
        assert!(!r.errored);
    }
}
