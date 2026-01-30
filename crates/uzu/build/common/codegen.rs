use std::{ffi::OsStr, fs};

use anyhow::Context;
use proc_macro2::TokenStream;

pub fn write_tokens(
    tokens: impl Into<TokenStream>,
    file: impl AsRef<OsStr>,
) -> anyhow::Result<()> {
    let tokens = tokens.into();
    let file = file.as_ref();

    let parsed = syn::parse2(tokens.into())
        .context("cannot parse generated bindings")?;
    fs::write(&file, prettyplease::unparse(&parsed))
        .with_context(|| format!("cannot write file {}", file.display()))?;

    std::process::Command::new("rustfmt")
        .arg(&file)
        .status()
        .context("rustfmt failed")?;

    Ok(())
}
