use std::{env, fs, path::PathBuf};
use anyhow::Context;
use tokio::try_join;

mod common;
mod metal;
mod shared_types;
mod vulkan;

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    debug_log!("build script started");

    if cfg!(feature = "build-clean") {
        let out_dir =
            PathBuf::from(env::var("OUT_DIR").context("missing OUT_DIR")?);
        if out_dir.exists() {
            fs::remove_dir_all(&out_dir).with_context(|| {
                format!("cannot clean {}", out_dir.display())
            })?;
            fs::create_dir_all(&out_dir).with_context(|| {
                format!("cannot recreate {}", out_dir.display())
            })?;
        }
        debug_log!("cleaned caches");
    }

    try_join!(metal::main(), shared_types::main())?;
    if cfg!(feature = "vulkan") {
        vulkan::main().await;
    }

    debug_log!("build script ended");

    Ok(())
}
