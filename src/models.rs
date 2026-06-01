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
    let env_var_for = |name: &str| -> &str {
        match name {
            "openai" => "OPENAI_API_KEY",
            "groq" => "GROQ_API_KEY",
            "anthropic" => "ANTHROPIC_API_KEY",
            "mistral" => "MISTRAL_API_KEY",
            "together" => "TOGETHER_API_KEY",
            "openrouter" => "OPENROUTER_API_KEY",
            _ => "API_KEY",
        }
    };

    for p in &cloud {
        if p.configured {
            println!(
                "  {} {}  {} ✓",
                "●".green(),
                p.name.bold(),
                env_var_for(p.name).dimmed(),
            );
            for m in &p.models {
                println!("      {}  (auto-selected default)", m.dimmed());
            }
        } else {
            println!(
                "  {} {}  {} not set",
                "○".dimmed(),
                p.name.dimmed(),
                env_var_for(p.name).dimmed(),
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
