//! Asset source for SVG icons. GPUI's `svg().path("icons/foo.svg")` resolves
//! through the `AssetSource` registered via `application().with_assets`.
//!
//! Backed by the on-disk `assets/` directory (located via `CARGO_MANIFEST_DIR`).
//! In dev this means adding/editing icons needs no rebuild. For a shipped bundle
//! this should be swapped for an embedded source.

use std::{borrow::Cow, fs, io::ErrorKind, path::PathBuf};

use anyhow::Result;
use gpui::{AssetSource, SharedString};

pub struct Assets {
    root: PathBuf,
}

impl Assets {
    pub fn new() -> Self {
        Self {
            root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets"),
        }
    }
}

impl AssetSource for Assets {
    fn load(&self, path: &str) -> Result<Option<Cow<'static, [u8]>>> {
        match fs::read(self.root.join(path)) {
            Ok(bytes) => Ok(Some(Cow::Owned(bytes))),
            Err(err) if err.kind() == ErrorKind::NotFound => Ok(None),
            Err(err) => Err(err.into()),
        }
    }

    fn list(&self, path: &str) -> Result<Vec<SharedString>> {
        let mut out = Vec::new();
        if let Ok(entries) = fs::read_dir(self.root.join(path)) {
            let prefix = path.trim_end_matches('/');
            for entry in entries.flatten() {
                if let Some(name) = entry.file_name().to_str() {
                    out.push(SharedString::from(format!("{prefix}/{name}")));
                }
            }
        }
        Ok(out)
    }
}
