use std::{ffi::OsStr, fs, path::Path};

use anyhow::Context;
use proc_macro2::TokenStream;

pub fn write_tokens(
    tokens: impl Into<TokenStream>,
    file: impl AsRef<OsStr>,
) -> anyhow::Result<()> {
    let tokens = tokens.into();
    let file = file.as_ref();

    let parsed = syn::parse2(tokens.clone()).with_context(|| format!("cannot parse generated bindings: {}", tokens))?;
    let new_contents = prettyplease::unparse(&parsed);
    if Path::new(file).exists() && fs::read(file).is_ok_and(|old_contents| old_contents == new_contents.as_bytes()) {
        return Ok(());
    }
    fs::write(file, new_contents).with_context(|| format!("cannot write file {}", file.display()))?;

    Ok(())
}
