#[cfg(any(target_vendor = "apple", windows))]
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    use benchmarks_uzu::Args;
    use clap::Parser;

    benchmarks_uzu::run(Args::parse()).await
}

// Apple platforms read mach task info and run on Metal; Windows reads its process and DXGI counters and runs on
// Vulkan. Elsewhere the binary only says so, which keeps workspace-wide builds working on Linux.
#[cfg(not(any(target_vendor = "apple", windows)))]
fn main() {
    eprintln!("benchmarks-uzu runs on Apple platforms and Windows only");
    std::process::exit(1);
}
