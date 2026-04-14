use std::{env, fs, path::PathBuf};

use anyhow::Context;
use futures::future::try_join_all;

mod common;
use common::{compiler::Compiler, envs, gpu_types::GpuTypes, traitgen::traitgen_all};

mod cpu;

#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal;

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    println!("cargo::rerun-if-changed=.");

    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("ios") && env::var("IPHONEOS_DEPLOYMENT_TARGET").is_err() {
        println!("cargo::rustc-env=IPHONEOS_DEPLOYMENT_TARGET=26.0");
    }

    if envs::build_always() {
        println!("cargo::rerun-if-changed=/var/empty/hack_nonexistent_file_to_always_rerun");
    }

    let target_arch = env::var("CARGO_CFG_TARGET_ARCH")?;
    let target_os = env::var("CARGO_CFG_TARGET_OS")?;

    let metal_backend = cfg!(feature = "metal") && matches!(target_os.as_ref(), "macos" | "ios" | "tvos" | "visionos");
    println!("cargo::rustc-check-cfg=cfg(metal_backend)");
    if metal_backend {
        println!("cargo::rustc-cfg=metal_backend");
    }

    let grammar_xgrammar = cfg!(feature = "grammar") && target_arch != "wasm32";
    println!("cargo::rustc-check-cfg=cfg(grammar_xgrammar)");
    if grammar_xgrammar {
        println!("cargo::rustc-cfg=grammar_xgrammar");
    }

    debug_log!("build script started");

    if envs::build_clean() {
        let out_dir = PathBuf::from(env::var("OUT_DIR").context("missing OUT_DIR")?);
        if out_dir.exists() {
            fs::remove_dir_all(&out_dir).with_context(|| format!("cannot clean {}", out_dir.display()))?;
            fs::create_dir_all(&out_dir).with_context(|| format!("cannot recreate {}", out_dir.display()))?;
        }
        debug_log!("cleaned caches");
    }

    let gpu_types = GpuTypes::scan().context("Failed to scan gpu types")?;
    debug_log!("gpu_types scan done");

    let mut compilers: Vec<Box<dyn Compiler>> = Vec::new();

    compilers.push(Box::new(cpu::CpuCompiler::new()?));

    #[cfg(all(feature = "metal", target_os = "macos"))]
    if metal_backend {
        compilers.push(Box::new(metal::MetalCompiler::new()?));
    }

    let backends_kernels = try_join_all(compilers.iter().map(|c| c.build(&gpu_types))).await?;

    debug_log!("backend build end");

    traitgen_all(backends_kernels)?;

    debug_log!("build script ended");

    Ok(())
}
