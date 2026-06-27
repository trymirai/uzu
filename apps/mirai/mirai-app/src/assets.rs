//! `AssetSource` serving the SVG icons in `assets/`. `rust-embed` reads from
//! disk in debug builds (live icon reload) and embeds the bytes into the binary
//! in release, so a shipped `.app` bundle is self-contained (no dependency on
//! the source checkout's `assets/` dir at runtime). Fonts are embedded
//! separately via `include_bytes!` in `ui-kit/theme.rs`.

use std::borrow::Cow;

use anyhow::Result;
use gpui::{AssetSource, SharedString};

#[derive(rust_embed::RustEmbed)]
#[folder = "assets"]
struct Embedded;

pub struct Assets;

impl Assets {
    pub fn new() -> Self {
        Self
    }
}

impl AssetSource for Assets {
    fn load(
        &self,
        path: &str,
    ) -> Result<Option<Cow<'static, [u8]>>> {
        Ok(Embedded::get(path).map(|file| file.data))
    }

    fn list(
        &self,
        path: &str,
    ) -> Result<Vec<SharedString>> {
        Ok(Embedded::iter().filter(|p| p.starts_with(path)).map(|p| SharedString::from(p.to_string())).collect())
    }
}
