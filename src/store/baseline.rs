use super::Db;
use crate::cli::{BaselineAction, BaselineArgs};
use anyhow::Result;
use colored::Colorize;

pub async fn run(args: BaselineArgs) -> Result<()> {
    let db = Db::open()?;

    match args.action {
        BaselineAction::Set { run_id } => {
            let id = match run_id {
                Some(id) => id,
                None => db
                    .last_run_id()?
                    .ok_or_else(|| anyhow::anyhow!("No runs found. Run `crucible run` first."))?,
            };
            // Look up which suite this run belongs to so we scope the baseline correctly.
            let suite_name = db
                .suite_for_run(&id)?
                .ok_or_else(|| anyhow::anyhow!("Run '{}' not found in database.", id))?;
            db.set_baseline(&id, &suite_name)?;
            println!(
                "  {} Baseline set to run {} (suite: {})",
                "✓".green(),
                id.yellow(),
                suite_name.cyan()
            );
        }

        BaselineAction::Show => {
            let baselines = db.all_baselines()?;
            if baselines.is_empty() {
                println!(
                    "  {}",
                    "No baseline set. Run `crucible baseline set` after a passing run.".yellow()
                );
            } else {
                println!("{}", "  Current baselines:".bold());
                for (suite, b) in &baselines {
                    println!("  {:40} {} ({})", suite.cyan(), b.id.yellow(), b.timestamp);
                }
            }
        }
    }
    Ok(())
}
