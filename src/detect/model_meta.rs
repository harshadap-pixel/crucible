use crate::providers::ollama::ModelInfo;

#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone)]
pub enum AttentionArch {
    Dense,
    MHA, // head_count_kv == head_count
    GQA, // head_count_kv < head_count (grouped query)
    MQA, // head_count_kv == 1 (multi-query)
}

#[derive(Debug, Clone)]
pub struct MoeConfig {
    pub total_experts: u32,
    pub active_experts: u32,
}

#[derive(Debug, Clone)]
pub struct ModelMetadata {
    pub general_arch: String,
    pub attention: AttentionArch,
    pub moe: Option<MoeConfig>,
    pub context_length: u32,
    pub embedding_dim: u32,
    pub head_count: u32,
    pub head_count_kv: u32,
    pub parameter_size: String,
    pub quantization: String,
}

impl ModelMetadata {
    pub fn from_ollama(info: &ModelInfo) -> Self {
        let mi = &info.modelinfo;

        let get_u32 = |key: &str| -> u32 {
            mi.get(key)
                .and_then(|v| v.as_f64())
                .map(|f| f as u32)
                .unwrap_or(0)
        };

        let get_str = |key: &str| -> String {
            mi.get(key)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string()
        };

        let head_count = get_u32("llama.attention.head_count");
        let head_count_kv = get_u32("llama.attention.head_count_kv");
        let expert_count = get_u32("llama.expert_count");
        let expert_used = get_u32("llama.expert_used_count");
        let arch = get_str("general.architecture");

        // Detect MoE: explicit field OR architecture name
        let is_moe = expert_count > 0
            || ["mixtral", "qwen2_moe", "deepseek2", "moe"]
                .iter()
                .any(|a| arch.to_lowercase().contains(a));

        let moe = if is_moe && expert_count > 0 {
            Some(MoeConfig {
                total_experts: expert_count,
                active_experts: if expert_used > 0 { expert_used } else { 2 },
            })
        } else if is_moe {
            // Name-based detection — architecture says MoE but fields absent
            Some(MoeConfig {
                total_experts: 0,
                active_experts: 0,
            })
        } else {
            None
        };

        let attention = if head_count == 0 {
            AttentionArch::Dense
        } else if head_count_kv == 1 {
            AttentionArch::MQA
        } else if head_count_kv < head_count {
            AttentionArch::GQA
        } else {
            AttentionArch::MHA
        };

        Self {
            general_arch: if arch.is_empty() {
                "unknown".into()
            } else {
                arch
            },
            attention,
            moe,
            context_length: get_u32("llama.context_length"),
            embedding_dim: get_u32("llama.embedding_length"),
            head_count,
            head_count_kv,
            parameter_size: info.details.parameter_size.clone(),
            quantization: info.details.quantization_level.clone(),
        }
    }

    /// Recommended n_runs multiplier for reproducibility
    pub fn n_runs_multiplier(&self) -> u32 {
        // MoE router stochasticity → double sample count
        if self.moe.is_some() {
            2
        } else {
            1
        }
    }
}
