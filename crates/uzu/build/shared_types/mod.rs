use std::{env, fs, path::PathBuf};

use crate::debug_log;
use anyhow::Context;
use tokio::{task::spawn_blocking, try_join};

struct BindgenConfig {
    name: &'static str,
    kernel_subdir: &'static str,
    header_file: &'static str,
    output_file: &'static str,
    allowlist_types: &'static [&'static str],
    raw_lines: &'static [&'static str],
}

const MATMUL_CONFIG: BindgenConfig = BindgenConfig {
    name: "matmul",
    kernel_subdir: "matmul/common",
    header_file: "shared_types.h",
    output_file: "shared_types.rs",
    allowlist_types: &[
        "GEMMParams",
        "GEMMAddMMParams",
        "GEMMSpiltKParams",
        "GEMMSpiltKMlpFusedParams",
    ],
    raw_lines: &["#![allow(non_snake_case)]"],
};

const ATTENTION_CONFIG: BindgenConfig = BindgenConfig {
    name: "attention",
    kernel_subdir: "attention",
    header_file: "gemm_types.h",
    output_file: "gemm_types.rs",
    allowlist_types: &["AttnParams", "AttnMaskParams"],
    raw_lines: &[],
};

fn autogen_shared_types(config: &BindgenConfig) -> anyhow::Result<()> {
    let out_dir = env::var("OUT_DIR").context("missing OUT_DIR")?;
    let build_dir = PathBuf::from(out_dir).join("shared_types");
    fs::create_dir_all(&build_dir).with_context(|| {
        format!("cannot create shared types build dir {}", build_dir.display())
    })?;

    let manifest_dir = PathBuf::from(
        env::var("CARGO_MANIFEST_DIR").context("missing CARGO_MANIFEST_DIR")?,
    );
    let kernel_dir = manifest_dir
        .join(format!("src/backends/metal/kernel/{}", config.kernel_subdir));
    let types_header = kernel_dir.join(config.header_file);
    let types_header_hash: [u8; blake3::OUT_LEN] =
        blake3::hash(&fs::read(types_header.as_path()).with_context(|| {
            format!("cannot read {}", types_header.display())
        })?)
        .into();
    let cached_types_header_hash =
        build_dir.join(format!("{}.hash", config.header_file));

    if let Some(cached_hash) = fs::read(&cached_types_header_hash).ok()
        && cached_hash == types_header_hash
    {
        debug_log!("{} bindgen cached", config.name);
        return Ok(());
    }
    debug_log!("{} bindgen started", config.name);

    let mut builder = bindgen::Builder::default()
        .header(types_header.to_string_lossy())
        .clang_arg("-x")
        .clang_arg("c")
        .derive_default(true)
        .derive_copy(true)
        .use_core();

    for raw_line in config.raw_lines {
        builder = builder.raw_line(*raw_line);
    }

    for ty in config.allowlist_types {
        builder = builder.allowlist_type(*ty);
    }

    let bindings = builder.generate().map_err(|e| {
        anyhow::anyhow!(
            "failed to generate bindings from {}: {e}",
            config.header_file
        )
    })?;

    let bindings_path = kernel_dir.join(config.output_file);
    bindings.write_to_file(&bindings_path).with_context(|| {
        format!("failed to write {}", bindings_path.display())
    })?;

    fs::write(&cached_types_header_hash, &types_header_hash).with_context(
        || format!("failed to write {}", cached_types_header_hash.display()),
    )?;
    debug_log!("{} bindgen done", config.name);

    Ok(())
}

pub async fn main() -> anyhow::Result<()> {
    let (matmul_result, attn_result) = try_join!(
        spawn_blocking(|| autogen_shared_types(&MATMUL_CONFIG)),
        spawn_blocking(|| autogen_shared_types(&ATTENTION_CONFIG))
    )?;

    matmul_result?;
    attn_result?;

    Ok(())
}
