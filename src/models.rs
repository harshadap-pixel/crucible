use anyhow::Result;
use colored::Colorize;

use crate::providers;

pub async fn run(ollama_url: &str) -> Result<()> {
    println!("\n{}", "AVAILABLE PROVIDERS".bold().cyan());
    println!("{}", "─".repeat(62).dimmed());

    let providers = providers::detect_available(ollama_url).await;

    // ── Local ─────────────────────────────────────────────────────────────────
    println!("\n  {}", "LOCAL".bold());

    let ollama = providers.iter().find(|p| p.name == "ollama").unwrap();
    if ollama.configured {
        println!(
            "  {} {}  ({} model(s))",
            "●".green(),
            "ollama".bold(),
            ollama.models.len()
        );
        for m in &ollama.models {
            println!("      {}", m.dimmed());
        }
    } else {
        println!(
            "  {} {}  {}",
            "○".dimmed(),
            "ollama".dimmed(),
            "not running — start with: ollama serve".dimmed()
        );
    }

    // ── Cloud ─────────────────────────────────────────────────────────────────
    println!("\n  {}", "CLOUD".bold());

    let cloud: Vec<_> = providers.iter().filter(|p| p.name != "ollama").collect();
    for p in &cloud {
        if p.name == "azure" {
            if p.configured {
                println!(
                    "  {} {}  AZURE_OPENAI_API_KEY ✓  AZURE_OPENAI_ENDPOINT ✓",
                    "●".green(),
                    "azure".bold(),
                );
                println!(
                    "      {}",
                    "Use --model azure:<deployment>  e.g. azure:gpt-4o".dimmed()
                );
            } else {
                let key_ok = std::env::var("AZURE_OPENAI_API_KEY")
                    .map(|v| !v.is_empty())
                    .unwrap_or(false);
                let ep_ok = std::env::var("AZURE_OPENAI_ENDPOINT")
                    .map(|v| !v.is_empty())
                    .unwrap_or(false);
                println!(
                    "  {} {}  AZURE_OPENAI_API_KEY {}  AZURE_OPENAI_ENDPOINT {}",
                    "○".dimmed(),
                    "azure".dimmed(),
                    if key_ok {
                        "✓".green()
                    } else {
                        "not set".red()
                    },
                    if ep_ok {
                        "✓".green()
                    } else {
                        "not set".red()
                    },
                );
            }
            continue;
        }

        if p.name == "bedrock" {
            if p.configured {
                let region = std::env::var("AWS_DEFAULT_REGION")
                    .or_else(|_| std::env::var("AWS_REGION"))
                    .unwrap_or_else(|_| "us-east-1".to_string());
                println!(
                    "  {} {}  AWS_ACCESS_KEY_ID ✓  AWS_SECRET_ACCESS_KEY ✓  region={}",
                    "●".green(),
                    "bedrock".bold(),
                    region,
                );
                println!(
                    "      {}",
                    "Use --model bedrock:<model-id>  e.g. bedrock:anthropic.claude-3-5-sonnet-20241022-v2:0".dimmed()
                );
            } else {
                let key_id_ok = std::env::var("AWS_ACCESS_KEY_ID")
                    .map(|v| !v.is_empty())
                    .unwrap_or(false);
                let secret_ok = std::env::var("AWS_SECRET_ACCESS_KEY")
                    .map(|v| !v.is_empty())
                    .unwrap_or(false);
                println!(
                    "  {} {}  AWS_ACCESS_KEY_ID {}  AWS_SECRET_ACCESS_KEY {}",
                    "○".dimmed(),
                    "bedrock".dimmed(),
                    if key_id_ok {
                        "✓".green()
                    } else {
                        "not set".red()
                    },
                    if secret_ok {
                        "✓".green()
                    } else {
                        "not set".red()
                    },
                );
            }
            continue;
        }

        let env_key = match p.name {
            "openai" => "OPENAI_API_KEY",
            "groq" => "GROQ_API_KEY",
            "anthropic" => "ANTHROPIC_API_KEY",
            "mistral" => "MISTRAL_API_KEY",
            "together" => "TOGETHER_API_KEY",
            "openrouter" => "OPENROUTER_API_KEY",
            _ => "API_KEY",
        };
        if p.configured {
            println!(
                "  {} {}  {} ✓",
                "●".green(),
                p.name.bold(),
                env_key.dimmed(),
            );
            for m in &p.models {
                println!("      {}  (auto-selected default)", m.dimmed());
            }
        } else {
            println!(
                "  {} {}  {} not set",
                "○".dimmed(),
                p.name.dimmed(),
                env_key.dimmed(),
            );
        }
    }

    // ── Auto-select summary ───────────────────────────────────────────────────
    println!("\n{}", "─".repeat(62).dimmed());
    match providers::auto_select_model(&providers) {
        Some(model) => {
            println!(
                "  {} Auto-select would pick: {}\n",
                "→".green(),
                model.yellow().bold()
            );
            println!("  Run without --model and crucible will use this automatically.");
        }
        None => {
            println!("  {} Nothing configured.", "⚠".yellow());
            println!("  Start Ollama  →  ollama serve && ollama pull llama3.1:8b");
            println!("  Or set a key  →  export OPENAI_API_KEY=sk-...");
        }
    }
    println!();

    Ok(())
}
