use std::{env, fs, path::PathBuf};

use anyhow::Context;
use futures::future::try_join_all;

mod common;
use common::{codegen::write_tokens, compiler::Compiler, envs, gpu_types, traitgen::traitgen_all};
use igata::{enum_paths::EnumPaths, gpu_types::tile_geometry_tokens};

mod cpu;

#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal;

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    println!("cargo::rerun-if-changed=.");

    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("ios") && env::var("IPHONEOS_DEPLOYMENT_TARGET").is_err() {
        println!("cargo::rustc-env=IPHONEOS_DEPLOYMENT_TARGET=26.4");
    }

    if envs::build_always() {
        println!("cargo::rerun-if-changed=/var/empty/hack_nonexistent_file_to_always_rerun");
    }

    let target_os = env::var("CARGO_CFG_TARGET_OS")?;

    let metal_backend = cfg!(feature = "metal") && matches!(target_os.as_ref(), "macos" | "ios" | "tvos" | "visionos");
    println!("cargo::rustc-check-cfg=cfg(metal_backend)");
    if metal_backend {
        println!("cargo::rustc-cfg=metal_backend");
    }

    let grammar_xgrammar = cfg!(feature = "grammar");
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

    let gpu_types = gpu_types::scan().context("Failed to scan gpu types")?;
    debug_log!("gpu_types scan done");

    let enum_paths = EnumPaths::from_gpu_types(&gpu_types).context("Failed to build enum path map")?;

    let tile_geometry_path = PathBuf::from(env::var("OUT_DIR").context("missing OUT_DIR")?).join("tile_geometry.rs");
    write_tokens(tile_geometry_tokens(&gpu_types), &tile_geometry_path).context("cannot write tile geometry")?;

    let compilers: Vec<Box<dyn Compiler>> = vec![
        Box::new(cpu::CpuCompiler::new()?),
        #[cfg(all(feature = "metal", target_os = "macos"))]
        Box::new(metal::MetalCompiler::new()?),
    ];

    let backends_kernels = try_join_all(compilers.iter().map(|c| c.build(&gpu_types, &enum_paths))).await?;

    debug_log!("backend build end");

    traitgen_all(backends_kernels)?;

    debug_log!("build script ended");

    Ok(())
}
