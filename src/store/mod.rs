pub mod baseline;
pub mod db;

use anyhow::Result;
use colored::Colorize;

pub use db::Db;

pub async fn status() -> Result<()> {
    let db = Db::open()?;

    let run_count = db.run_count()?;
    let baselines = db.all_baselines()?;

    println!("{}", "\nCRUCIBLE STATUS".bold().cyan());
    println!("{}", "─".repeat(40).dimmed());
    println!("  Runs stored      {}", run_count);

    if baselines.is_empty() {
        println!(
            "  Baseline         {}",
            "not set — run `crucible baseline set`".yellow()
        );
    } else {
        println!("  Baselines:");
        for (suite, b) in &baselines {
            println!(
                "    {:38} {} ({})",
                suite.cyan(),
                b.id.yellow(),
                b.timestamp
            );
        }
    }

    let db_path = db.path();
    println!("  Database         {}", db_path.dimmed());
    println!();
    Ok(())
}
