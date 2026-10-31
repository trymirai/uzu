#[cfg(target_vendor = "apple")]
mod bench;
#[cfg(target_vendor = "apple")]
mod common;
#[cfg(target_vendor = "apple")]
mod engine;
#[cfg(target_vendor = "apple")]
mod memory;

#[cfg(target_vendor = "apple")]
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    use clap::Parser;

    use crate::{
        common::{StdoutRedirect, run_loop},
        engine::UzuEngine,
    };

    #[derive(Debug, Parser)]
    struct Args {
        #[arg(short, long)]
        model: String,
    }

    let args = Args::parse();
    let mut redirect = StdoutRedirect::to_stderr()?;
    let mut engine = UzuEngine::new(args.model.as_str()).await?;
    redirect.restore()?;
    run_loop(&mut engine).await?;
    Ok(())
}

// Apple platforms only: the memory counters read mach task info and the engine runs on Metal. Elsewhere the binary
// only says so, which keeps workspace-wide builds working on Windows and Linux.
#[cfg(not(target_vendor = "apple"))]
fn main() {
    eprintln!("benchmarks-uzu runs on Apple platforms only");
    std::process::exit(1);
}
