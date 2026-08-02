use std::{env, fs, path::PathBuf, process::ExitCode};

use anyhow::Context;

mod common;
use common::{compiler::Compiler, envs, gpu_type_paths::GpuTypePaths, gpu_types::GpuTypes, traitgen::traitgen_all};

#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal;

#[cfg(feature = "webgpu")]
mod slang;

#[cfg(feature = "webgpu")]
mod webgpu;

mod cpu;

fn main() -> anyhow::Result<ExitCode> {
    println!("cargo::rerun-if-changed=build");

    if envs::build_always() {
        println!("cargo::rerun-if-changed=/var/empty/hack_nonexistent_file_to_always_rerun");
    }

    let target_arch = env::var("CARGO_CFG_TARGET_ARCH")?;
    let target_os = env::var("CARGO_CFG_TARGET_OS")?;

    println!("cargo::rustc-check-cfg=cfg(backend, values(\"metal\", \"webgpu\", \"cpu\"))");

    let backend_metal = cfg!(feature = "metal") && matches!(target_os.as_ref(), "macos" | "ios" | "tvos" | "visionos");
    if backend_metal {
        println!("cargo::rustc-cfg=backend=\"metal\"");
    }

    let backend_webgpu = cfg!(feature = "webgpu");
    if backend_webgpu {
        println!("cargo::rustc-cfg=backend=\"webgpu\"");
    }

    let backend_cpu = cfg!(feature = "cpu");
    if backend_cpu {
        println!("cargo::rustc-cfg=backend=\"cpu\"");
    }

    let grammar = cfg!(feature = "grammar") && target_arch != "wasm32";
    println!("cargo::rustc-check-cfg=cfg(grammar)");
    if grammar {
        println!("cargo::rustc-cfg=grammar");
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").context("missing OUT_DIR")?);
    debug_log!("build script started in {}", out_dir.display());

    if envs::build_clean() {
        if out_dir.exists() {
            fs::remove_dir_all(&out_dir).with_context(|| format!("cannot clean {}", out_dir.display()))?;
            fs::create_dir_all(&out_dir).with_context(|| format!("cannot recreate {}", out_dir.display()))?;
        }
        debug_log!("cleaned caches");
    }

    let gpu_types = GpuTypes::scan("common").context("Failed to scan gpu types")?;
    debug_log!("gpu_types scan done");

    let gpu_type_paths = GpuTypePaths::from_gpu_types(&gpu_types).context("Failed to build GPU type path map")?;

    let mut compilers: Vec<Box<dyn Compiler>> = Vec::new();

    #[cfg(all(feature = "metal", target_os = "macos"))]
    if backend_metal {
        compilers.push(Box::new(metal::MetalCompiler::new()?));
    }

    #[cfg(feature = "webgpu")]
    if backend_webgpu {
        compilers.push(Box::new(slang::SlangCompiler::<webgpu::WebGPU>::new()?));
    }

    if backend_cpu {
        compilers.push(Box::new(cpu::CpuCompiler::new()?));
    }

    if compilers.is_empty() {
        println!("cargo::error=uzu requires at least one backend to be compiled in!");
        return Ok(ExitCode::FAILURE);
    }

    let backends_kernels =
        compilers.iter().map(|c| c.build(&gpu_types, &gpu_type_paths)).collect::<anyhow::Result<Vec<_>>>()?;

    debug_log!("backend build end");

    traitgen_all(backends_kernels, &gpu_type_paths)?;

    debug_log!("build script ended");

    Ok(ExitCode::SUCCESS)
}
