use std::{env, fs, path::PathBuf};

use anyhow::Context;
use compiler::{MetalCompiler, ObjectInfo};
use futures::{StreamExt, TryStreamExt, stream};
use proc_macro2::TokenStream;
use quote::quote;
use walkdir::WalkDir;

mod ast;
mod bindgen;
mod compiler;
mod toolchain;

pub async fn main() -> anyhow::Result<()> {
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

    if !all_cached {
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
    }

    Ok(())
}
