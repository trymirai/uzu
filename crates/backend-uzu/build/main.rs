use std::{env, fs, path::PathBuf, process::ExitCode};

use anyhow::Context;
use futures::future::try_join_all;

mod common;
use common::{compiler::Compiler, enum_paths::EnumPaths, envs, gpu_types::GpuTypes, traitgen::traitgen_all};

mod cpu;

#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal;

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<ExitCode> {
    println!("cargo::rerun-if-changed=.");
    if envs::build_always() {
        println!("cargo::rerun-if-changed=/var/empty/hack_nonexistent_file_to_always_rerun");
    }

    let target_arch = env::var("CARGO_CFG_TARGET_ARCH")?;
    let target_os = env::var("CARGO_CFG_TARGET_OS")?;

    println!("cargo::rustc-check-cfg=cfg(backend, values(\"cpu\", \"metal\"))");

    let backend_cpu = cfg!(feature = "cpu");
    if backend_cpu {
        println!("cargo::rustc-cfg=backend=\"cpu\"");
    }

    let backend_metal = cfg!(feature = "metal") && matches!(target_os.as_ref(), "macos" | "ios" | "tvos" | "visionos");
    if backend_metal {
        println!("cargo::rustc-cfg=backend=\"metal\"");
    }

    let grammar = cfg!(feature = "grammar") && target_arch != "wasm32";
    println!("cargo::rustc-check-cfg=cfg(grammar)");
    if grammar {
        println!("cargo::rustc-cfg=grammar");
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

    let enum_paths = EnumPaths::from_gpu_types(&gpu_types).context("Failed to build enum path map")?;

    #[allow(unused_mut)]
    let mut compilers: Vec<Box<dyn Compiler>> = Vec::new();

    if backend_cpu {
        compilers.push(Box::new(cpu::CpuCompiler::new()?));
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    if backend_metal {
        compilers.push(Box::new(metal::MetalCompiler::new()?));
    }

    if compilers.is_empty() {
        println!("cargo::error=uzu requires at least one backend to be compiled in!");
        return Ok(ExitCode::FAILURE);
    }

    let backends_kernels = try_join_all(compilers.iter().map(|c| c.build(&gpu_types, &enum_paths))).await?;

    debug_log!("backend build end");

    traitgen_all(backends_kernels)?;

    debug_log!("build script ended");

    Ok(ExitCode::SUCCESS)
}
