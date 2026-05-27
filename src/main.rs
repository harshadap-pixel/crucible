use anyhow::Result;
use clap::Parser;
use cli::{Cli, Command};
use crucible::{cli, detect, discover, regression, report, runner, store};

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Run(args) => runner::run(args).await,
        Command::Detect(args) => detect::run(args).await,
        Command::Baseline(args) => store::baseline::run(args).await,
        Command::Compare(args) => regression::compare(args).await,
        Command::Report(args) => report::history(args).await,
        Command::Status => store::status().await,
        Command::Autodiscover(args) => discover::run(args).await,
    }
}
