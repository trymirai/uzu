#[cfg(target_vendor = "apple")]
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    use benchmarks_uzu::Args;
    use clap::Parser;

    benchmarks_uzu::run(Args::parse()).await
}

// Apple platforms only: the memory counters read mach task info and the engine runs on Metal. Elsewhere the binary
// only says so, which keeps workspace-wide builds working on Windows and Linux.
#[cfg(not(target_vendor = "apple"))]
fn main() {
    eprintln!("benchmarks-uzu runs on Apple platforms only");
    std::process::exit(1);
}
