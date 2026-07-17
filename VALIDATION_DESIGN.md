# Judge Validation Harness — Design Document

## Overview
Validate that Crucible's LLM judge produces consistent, reliable evaluations across multiple judge models.

## Problem Statement
Currently, the judge has no self-validation. We can't prove:
- Precision/recall of failure detection
- Agreement with human judgment
- Consistency across rubrics

## Solution Architecture

### Data Flow
```
User Input
  ↓
[1] Load Suite + Select Model
  ↓
[2] Execute Suite (run tests → capture outputs)
  ↓
[3] Extract llm_judge Assertions
  ↓
[4] Judge Evaluation (run each output through N judges)
  ├─ Judge A scores output on rubric → 0.85
  ├─ Judge B scores output on rubric → 0.82
  └─ Judge C scores output on rubric → 0.88
  ↓
[5] Compute Metrics
  ├─ Agreement % (how close are scores?)
  ├─ Disagreement cases (>0.15 gap)
  └─ Per-rubric confidence
  ↓
[6] Generate Report
  └─ Overall reliability + recommendations
```

## Key Components

### 1. Judge Response Collector
**Purpose:** Run output through judges WITHOUT comparing to threshold yet

**Input:** 
- Test output (actual model response)
- Rubric (evaluation criteria)
- Judge model spec

**Output:**
- Score (0.0-1.0)
- Reason (judge explanation)
- Judge name

**Design:** Create a separate function from llm_judge.rs that just returns the raw score

### 2. Agreement Metrics
**Metrics to compute:**

1. **Percentage Agreement** (simple)
   - % of cases where all judges agree within 0.15 score margin
   - Formula: (agreements / total_cases) * 100

2. **Max Disagreement** (per case)
   - For each test, max_score - min_score
   - Flag if > 0.15

3. **Per-Rubric Reliability**
   - Agreement % broken down by rubric
   - Low agreement rubrics need clarification

**Do NOT compute:**
- Cohen's kappa (too complex for MVP)
- Krippendorff's alpha (overkill for now)
- Keep it simple and interpretable

### 3. Data Structures
```rust
struct JudgeEvaluation {
    test_name: String,
    output: String,           // Actual model output
    rubric: String,
    judge_scores: {
        judge_name: f64,      // Score from each judge
    },
    agreement_pct: f64,
    divergent: bool,          // true if max_disagreement > 0.15
}

struct ValidationReport {
    test_cases: Vec<JudgeEvaluation>,
    overall_agreement: f64,
    per_rubric_stats: HashMap<String, RubricStats>,
    divergent_cases: Vec<JudgeEvaluation>,
    confidence_level: String,  // high/medium/low
}
```

## Implementation Phases

### Phase 1: Integration with Test Runner ✓ (design)
- Reuse `execute_suite()` to run tests
- Capture TestResult.output for each test
- Extract llm_judge assertions

### Phase 2: Judge Evaluation
- Create `score_output_with_judge()` function
- Run each output through each judge model
- Collect raw scores (no pass/fail logic yet)
- Handle errors gracefully (log failures, continue)

### Phase 3: Metrics Computation
- Compute agreement percentages
- Identify divergent cases
- Group by rubric
- Generate confidence level

### Phase 4: Reporting
- Pretty-print results
- Show problematic rubrics
- Recommend rubric clarifications
- Output JSON for CI integration

### Phase 5: Testing
- Unit tests for metrics computation
- Integration test with sample suite
- Test error handling (judge timeouts, API failures)

## Testing Strategy

### Unit Tests
```rust
#[test]
fn test_agreement_calculation() {
    // Test: 3 judges all agree on 0.85 → 100% agreement
    // Test: judges score [0.85, 0.70] → divergent
}

#[test]
fn test_per_rubric_metrics() {
    // Test: rubric A has 90% agreement, B has 70%
}
```

### Integration Tests
```rust
#[tokio::test]
async fn test_validate_default_suite() {
    // Run validation on default.toml (small suite)
    // Should not require real API keys (mock judges)
}
```

## CLI Usage

### Current (before implementation)
```bash
crucible validate-judge --suite safety/owasp_llm01_injection.toml
```

### Proper (after implementation)
```bash
# Validate with default judges (Fable, Haiku, Groq)
crucible validate-judge --suite safety/owasp_llm01_injection.toml \
  --model llama3.1:8b

# Custom judges
crucible validate-judge --suite default.toml \
  --model gpt-4o-mini \
  --judges "anthropic:claude-fable-5,groq:llama-3.1-8b-instant"

# Show divergent cases
crucible validate-judge --show-divergent
```

## Success Criteria

- [ ] Runs a small test suite without errors
- [ ] Collects scores from multiple judges on same outputs
- [ ] Computes agreement metrics correctly
- [ ] Identifies divergent cases accurately
- [ ] Generates readable report
- [ ] Tests pass (unit + integration)
- [ ] CI passes (formatting, clippy, tests)
- [ ] Documentation complete

## Known Limitations (MVP)
- No human ground truth (can't measure against human judgment)
- Only supports llm_judge assertions
- Requires API keys for cloud judges
- Limited to one test suite per run (can be extended later)

## Future Enhancements
- Support for human-labeled ground truth
- Statistical significance testing
- Regression detection (when judge reliability drops)
- Per-model judge reliability tracking
- Automated rubric improvement suggestions
