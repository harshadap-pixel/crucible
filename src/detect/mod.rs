pub mod model_meta;
pub mod pipeline;

use crate::cli::DetectArgs;
use crate::providers::OllamaClient;
use anyhow::Result;
use colored::Colorize;
use comfy_table::Table;
use model_meta::{AttentionArch, ModelMetadata};

pub async fn run(args: DetectArgs) -> Result<()> {
    let client = OllamaClient::new(&args.ollama_url);

    if !client.health().await {
        anyhow::bail!(
            "Ollama not reachable at {}. Is it running?",
            args.ollama_url
        );
    }

    println!("{}", "\nDETECTING MODEL MECHANISMS".bold().cyan());
    println!("{}", "─".repeat(62).dimmed());

    // ── Model metadata ────────────────────────────────────────────
    print!("  Fetching model info for {}... ", args.model.yellow());
    let info = client.show(&args.model).await?;
    let meta = ModelMetadata::from_ollama(&info);
    println!("{}", "done".green());

    // ── KV cache probe ────────────────────────────────────────────
    let kv = if args.probe_cache {
        print!("  Probing KV cache (2 inference calls)... ");
        let r = client.probe_kv_cache(&args.model).await?;
        println!("{}", "done".green());
        Some(r)
    } else {
        None
    };

    // ── Pipeline config static scan ───────────────────────────────
    let pipeline = if let Some(ref path) = args.pipeline_config {
        Some(pipeline::scan(path)?)
    } else {
        None
    };

    // ── Print report ──────────────────────────────────────────────
    println!();
    println!("{}", "MODEL METADATA".bold());
    println!("{}", "─".repeat(62).dimmed());

    let mut table = Table::new();
    table.load_preset(comfy_table::presets::NOTHING);

    // Architecture + MoE
    let arch_label = match &meta.attention {
        AttentionArch::MHA => "MHA (Multi-Head Attention)",
        AttentionArch::GQA => "GQA (Grouped Query Attention)",
        AttentionArch::MQA => "MQA (Multi-Query Attention)",
        AttentionArch::Dense => "Dense",
    };

    table.add_row(vec!["Architecture", &meta.general_arch]);
    table.add_row(vec!["Attention type", arch_label]);

    if let Some(ref moe) = meta.moe {
        table.add_row(vec![
            "MoE",
            &format!(
                "YES — {} total experts / {} active per token ({:.1}% sparse)",
                moe.total_experts,
                moe.active_experts,
                (1.0 - moe.active_experts as f64 / moe.total_experts as f64) * 100.0
            ),
        ]);
    } else {
        table.add_row(vec!["MoE", "No (dense model)"]);
    }

    table.add_row(vec![
        "Context length",
        &format!("{} tokens", meta.context_length),
    ]);
    table.add_row(vec!["Embedding dim", &format!("{}", meta.embedding_dim)]);
    table.add_row(vec!["Q heads", &format!("{}", meta.head_count)]);
    table.add_row(vec!["KV heads", &format!("{}", meta.head_count_kv)]);
    table.add_row(vec!["Parameters", &meta.parameter_size]);
    table.add_row(vec!["Quantization", &meta.quantization]);

    println!("{table}");

    // KV cache probe result
    if let Some(ref kv) = kv {
        println!();
        println!("{}", "KV CACHE".bold());
        println!("{}", "─".repeat(62).dimmed());
        println!(
            "  Cold latency  {}ms   Warm latency  {}ms   Speedup  {:.2}x",
            kv.cold_latency_ms, kv.warm_latency_ms, kv.speedup_ratio
        );
        if kv.likely_active {
            println!("  Status  {}", "LIKELY ACTIVE".green().bold());
        } else {
            println!(
                "  Status  {}",
                "NOT DETECTED (or no shared prefix benefit)".yellow()
            );
        }
    }

    // Pipeline config scan
    if let Some(ref p) = pipeline {
        println!();
        println!("{}", "PIPELINE MECHANISMS (from config)".bold());
        println!("{}", "─".repeat(62).dimmed());
        for (k, v) in &p.signals {
            let status = if *v {
                "DETECTED".green().bold().to_string()
            } else {
                "not found".dimmed().to_string()
            };
            println!("  {:<30} {}", k, status);
        }
    }

    // MoE reproducibility warning
    if meta.moe.is_some() {
        println!();
        println!(
            "  {} MoE model detected — recommend --n-runs 5+ for stable scores",
            "⚠".yellow()
        );
    }

    println!();
    Ok(())
}
