//! Compile the browser kernels from Slang at build time, as the Vulkan backend
//! does for SPIR-V. The browser only receives the resulting WGSL.
use std::{
    collections::hash_map::DefaultHasher,
    env, fs,
    hash::{Hash, Hasher},
    path::PathBuf,
    process::Command,
};

use anyhow::{Context, bail};

pub fn compile() -> anyhow::Result<()> {
    let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?).join("src/backends/webgpu/kernel/slang");
    let output = PathBuf::from(env::var("OUT_DIR")?).join("webgpu_slang");
    fs::create_dir_all(&output)?;
    println!("cargo::rerun-if-changed={}", root.display());
    println!("cargo::rerun-if-env-changed=SLANGC");
    println!("cargo::rerun-if-env-changed=SLANG_DIR");
    let compiler = env::var_os("SLANGC").map(PathBuf::from).unwrap_or_else(|| {
        env::var_os("SLANG_DIR").map_or_else(|| PathBuf::from("slangc"), |dir| PathBuf::from(dir).join("bin/slangc"))
    });
    let version = Command::new(&compiler).arg("-version").output().with_context(|| {
        "WebGPU kernels require Slang 2026.19. Install the official compiler and set SLANGC to its executable, or SLANG_DIR to its installation directory"
    })?;
    if !version.status.success() {
        bail!("Could not run {} -version", compiler.display());
    }
    let version = format!("{}{}", String::from_utf8_lossy(&version.stdout), String::from_utf8_lossy(&version.stderr));
    if version.trim() != "2026.19" {
        bail!("WebGPU kernels are validated with Slang 2026.19, found {}", version.trim());
    }
    let abi = fs::read(root.join("abi.slang"))?;
    let mut sources = fs::read_dir(&root)?.collect::<Result<Vec<_>, _>>()?;
    sources.sort_by_key(|entry| entry.path());
    let mut dispatch =
        String::from("pub(super) fn compiled_shader(name: &str, width: u32) -> &'static str { match (name, width) {\n");
    for entry in sources {
        let path = entry.path();
        if path.extension().is_none_or(|value| value != "slang") || path.file_stem().is_some_and(|value| value == "abi")
        {
            continue;
        }
        let name = path.file_stem().context("missing shader name")?.to_str().context("shader name is not UTF-8")?;
        let source = fs::read_to_string(&path)?;
        let widths: Vec<u32> = if let Some(values) =
            source.lines().find_map(|line| line.trim().strip_prefix("// uzu-widths:"))
        {
            let widths = values.split(',').map(|value| value.trim().parse::<u32>()).collect::<Result<Vec<_>, _>>()?;
            if widths.is_empty() || widths.iter().any(|&width| width == 0 || width > 256) {
                bail!("Invalid Slang width variants in {name}");
            }
            widths
        } else if source.contains("WIDTH") {
            vec![1, 2, 4, 8, 16, 32, 64, 128, 256]
        } else {
            vec![0]
        };
        for width in widths {
            let wgsl = output.join(format!("{name}_{width}.wgsl"));
            let stamp = wgsl.with_extension("hash");
            let mut hasher = DefaultHasher::new();
            (&abi, &source, &version, width, include_bytes!("webgpu_slang.rs")).hash(&mut hasher);
            let hash = hasher.finish().to_string();
            if !wgsl.exists() || fs::read_to_string(&stamp).ok().as_deref() != Some(&hash) {
                let result = Command::new(&compiler)
                    .arg(&path)
                    .args([
                        "-target",
                        "wgsl",
                        "-entry",
                        "main",
                        "-O3",
                        "-fp-mode",
                        "precise",
                        "-line-directive-mode",
                        "none",
                    ])
                    .arg(format!("-DWIDTH={}", width.max(1)))
                    .arg("-o")
                    .arg(&wgsl)
                    .output()?;
                if !result.status.success() {
                    bail!(
                        "Slang compilation failed for {name} ({width}):\n{}",
                        String::from_utf8_lossy(&result.stderr)
                    );
                }
                let compiled = fs::read_to_string(&wgsl)?;
                let mut linked = format!("// uzu-kernel: {name}\n");
                // These declarations describe the typed ABI only. Tensor layouts,
                // offsets and aliases are supplied by the existing backend at run time.
                for line in source.lines() {
                    for (prefix, kind) in [("UZU_PARAM(", "param"), ("UZU_ARRAY(", "array")] {
                        if let Some(body) = line.trim().strip_prefix(prefix).and_then(|body| body.strip_suffix(')')) {
                            let (ty, field) = body.split_once(',').context("invalid Slang ABI declaration")?;
                            linked.push_str(&format!("// uzu-{kind}: {} {}\n", ty.trim(), field.trim()));
                        }
                    }
                }
                if compiled.contains("@builtin(subgroup_id)") {
                    linked.push_str("requires subgroup_id;\n");
                }
                linked.push_str(&compiled);
                fs::write(&wgsl, linked)?;
                fs::write(&stamp, &hash)?;
            }
            dispatch.push_str(&format!(
                "({name:?}, {width}) => include_str!({:?}),\n",
                wgsl.to_str().context("shader path is not UTF-8")?
            ));
        }
    }
    dispatch.push_str("_ => panic!(\"unknown Slang WebGPU kernel {name} width {width}\"),\n}}\n");
    fs::write(output.parent().unwrap().join("webgpu_slang.rs"), dispatch)?;
    Ok(())
}
