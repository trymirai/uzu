use std::{fs, sync::OnceLock};

use anyhow::Context;

static BUILD_SYSTEM_HASH: OnceLock<blake3::Hash> = OnceLock::new();

pub fn build_system_hash() -> anyhow::Result<&'static blake3::Hash> {
    if let Some(bsh) = BUILD_SYSTEM_HASH.get() {
        Ok(bsh)
    } else {
        let bsh = blake3::hash(
            &fs::read(&std::env::current_exe().context("cannot get build system executable")?)
                .context("cannot read build system executable")?,
        );

        Ok(BUILD_SYSTEM_HASH.get_or_init(|| bsh))
    }
}
