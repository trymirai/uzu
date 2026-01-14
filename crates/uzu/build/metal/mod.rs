use std::{
    env, fs,
    path::{Path, PathBuf},
};

use anyhow::Context;
use compiler::{MetalCompiler, ObjectInfo};
use futures::{StreamExt, TryStreamExt, stream};
use proc_macro2::TokenStream;
use quote::quote;
use walkdir::WalkDir;

use crate::debug_log;

mod ast;
mod bindgen;
mod compiler;
mod toolchain;

fn is_nax_source(path: &Path) -> bool {
    path.file_name()
        .and_then(|s| s.to_str())
        .map(|s| s.contains("_nax"))
        .unwrap_or(false)
}

pub async fn main() -> anyhow::Result<()> {
    let out_dir =
        PathBuf::from(env::var("OUT_DIR").context("missing OUT_DIR")?);

    let nax_enabled = cfg!(feature = "metal-nax");
    debug_log!("metal nax enabled: {nax_enabled}");

    let nax_mode_path = out_dir.join("metal_nax_mode");
    let desired_nax_mode = if nax_enabled { "1" } else { "0" };
    let prev_nax_mode =
        fs::read_to_string(&nax_mode_path).ok().map(|s| s.trim().to_owned());
    let nax_mode_changed =
        prev_nax_mode.as_deref() != Some(desired_nax_mode);

    let dsl = MetalCompiler::new().context("cannot create metal dsl")?;

    let src_dir = PathBuf::from(
        env::var("CARGO_MANIFEST_DIR").context("missing CARGO_MANIFEST_DIR")?,
    )
    .join("src");

    let metal_sources: Vec<PathBuf> = WalkDir::new(&src_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.file_type().is_file()
                && e.path().extension().and_then(|s| s.to_str())
                    == Some("metal")
        })
        .map(|e| e.into_path())
        .filter(|p| nax_enabled || !is_nax_source(p))
        .collect();

    let num_concurrent_compiles =
        std::thread::available_parallelism().map(|x| x.get()).unwrap_or(4) * 2;

    let results: Vec<(ObjectInfo, bool)> = stream::iter(metal_sources)
        .map(|p| dsl.compile(p))
        .buffer_unordered(num_concurrent_compiles)
        .try_collect()
        .await
        .context("cannot compile metal sources")?;

    let all_cached = results.iter().all(|(_, cached)| *cached);
    let kernels: Vec<ObjectInfo> =
        results.into_iter().map(|(o, _)| o).collect();

    if !all_cached || nax_mode_changed {
        dsl.link(&kernels).await.context("cannot link objects")?;

        let bindings = kernels
            .iter()
            .flat_map(|k| &k.kernels)
            .map(|k| {
                bindgen::bindgen(k).with_context(|| {
                    format!("cannot generate bindings for {}", k.name)
                })
            })
            .collect::<anyhow::Result<Vec<TokenStream>>>()?;

        let tokens = quote! { #(#bindings)* };

        let out_path =
            PathBuf::from(env::var("OUT_DIR").context("missing OUT_DIR")?)
                .join("dsl.rs");
        let parsed =
            syn::parse2(tokens).context("cannot parse generated bindings")?;
        fs::write(&out_path, prettyplease::unparse(&parsed)).with_context(
            || format!("cannot write bindings file {}", out_path.display()),
        )?;

        fs::write(&nax_mode_path, desired_nax_mode).with_context(|| {
            format!(
                "cannot write nax mode file {}",
                nax_mode_path.display()
            )
        })?;
    }

    Ok(())
}
