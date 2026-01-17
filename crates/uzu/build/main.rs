use std::{env, fs, path::PathBuf};

use anyhow::Context;
use futures::future::try_join_all;

mod common;
mod shared_types;

#[cfg(feature = "metal")]
mod metal;

use common::compiler::Compiler;
use common::envs;

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    println!("cargo::rerun-if-changed=.");

    if envs::build_always() {
        println!(
            "cargo::rerun-if-changed=/var/empty/hack_nonexistent_file_to_always_rerun"
        );
    }

    debug_log!("build script started");

    if envs::build_clean() {
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

    let mut compilers: Vec<Box<dyn Compiler>> = Vec::new();

    compilers.push(Box::new(shared_types::SharedTypesCompiler::new()?));

    #[cfg(feature = "metal")]
    compilers.push(Box::new(metal::MetalCompiler::new()?));

    try_join_all(compilers.iter().map(|c| c.build())).await?;

    debug_log!("build script ended");

    Ok(())
}
