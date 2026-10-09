use std::{ffi::OsStr, fs, io::ErrorKind, path::Path};

use anyhow::Context;
use proc_macro2::TokenStream;

pub fn write_tokens(
    tokens: impl Into<TokenStream>,
    file: impl AsRef<OsStr>,
) -> anyhow::Result<()> {
    let tokens = tokens.into();
    let file = file.as_ref();

    let parsed = syn::parse2(tokens.clone()).with_context(|| format!("cannot parse generated bindings: {}", tokens))?;
    write_if_changed(Path::new(file), prettyplease::unparse(&parsed))
}

/// Writes generated `contents` to `file` unless it already holds exactly these bytes, so an unchanged output keeps its
/// modification time and an immediate rebuild rewrites nothing. Only a missing file is created; any other read or write
/// error is returned with its source.
pub fn write_if_changed(
    file: &Path,
    contents: impl AsRef<[u8]>,
) -> anyhow::Result<()> {
    let contents = contents.as_ref();
    match fs::read(file) {
        Ok(existing) if existing == contents => return Ok(()),
        Ok(_) => {},
        Err(error) if error.kind() == ErrorKind::NotFound => {},
        Err(error) => return Err(error).with_context(|| format!("cannot read file {}", file.display())),
    }
    fs::write(file, contents).with_context(|| format!("cannot write file {}", file.display()))
}
