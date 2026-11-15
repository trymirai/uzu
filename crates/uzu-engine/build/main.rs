use std::{env, fs, path::PathBuf, process::ExitCode};

use anyhow::Context;

mod common;
use common::{compiler::Compiler, envs, gpu_type_paths::GpuTypePaths, gpu_types::GpuTypes, traitgen::traitgen_all};

#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal;

#[cfg(any(feature = "webgpu", feature = "vulkan"))]
mod slang;

#[cfg(feature = "webgpu")]
mod webgpu;

#[cfg(feature = "vulkan")]
mod vulkan;

mod cpu;

fn main() -> anyhow::Result<ExitCode> {
    println!("cargo::rerun-if-changed=build");

    if envs::build_always() {
        println!("cargo::rerun-if-changed=/var/empty/hack_nonexistent_file_to_always_rerun");
    }

    let target_os = env::var("CARGO_CFG_TARGET_OS")?;

    println!("cargo::rustc-check-cfg=cfg(backend, values(\"metal\", \"webgpu\", \"vulkan\", \"cpu\"))");

    let backend_metal = cfg!(feature = "metal") && matches!(target_os.as_ref(), "macos" | "ios" | "tvos" | "visionos");
    if backend_metal {
        println!("cargo::rustc-cfg=backend=\"metal\"");
    }

    let backend_webgpu = cfg!(feature = "webgpu");
    if backend_webgpu {
        println!("cargo::rustc-cfg=backend=\"webgpu\"");
    }

    let backend_vulkan = cfg!(feature = "vulkan");
    if backend_vulkan {
        println!("cargo::rustc-cfg=backend=\"vulkan\"");
    }

    let backend_cpu = cfg!(feature = "cpu");
    if backend_cpu {
        println!("cargo::rustc-cfg=backend=\"cpu\"");
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

        let src_file = PathBuf::from("src/vision/metal_vision.m");
        if src_file.exists() {
            println!("cargo::rerun-if-changed={}", src_file.display());
            let obj_file = out_dir.join("metal_vision.o");
            let lib_file = out_dir.join("libmetal_vision.a");
            let clang_status = std::process::Command::new("clang")
                .args([
                    "-c",
                    "-O3",
                    src_file.to_str().unwrap(),
                    "-o",
                    obj_file.to_str().unwrap(),
                    "-fobjc-arc",
                ])
                .status()
                .context("Failed to execute clang for metal_vision.m")?;
            if !clang_status.success() {
                anyhow::bail!("clang failed to compile metal_vision.m");
            }

            let ar_status = std::process::Command::new("ar")
                .args(["crus", lib_file.to_str().unwrap(), obj_file.to_str().unwrap()])
                .status()
                .context("Failed to execute ar for libmetal_vision.a")?;
            if !ar_status.success() {
                anyhow::bail!("ar failed to create libmetal_vision.a");
            }

            println!("cargo::rustc-link-search=native={}", out_dir.display());
            println!("cargo::rustc-link-lib=static=metal_vision");
            println!("cargo::rustc-link-lib=framework=MetalPerformanceShaders");
        }
    }

    #[cfg(feature = "webgpu")]
    if backend_webgpu {
        compilers.push(Box::new(slang::SlangCompiler::<webgpu::WebGPU>::new()?));
    }

    #[cfg(feature = "vulkan")]
    if backend_vulkan {
        compilers.push(Box::new(slang::SlangCompiler::<vulkan::VulkanTarget>::new()?));
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
