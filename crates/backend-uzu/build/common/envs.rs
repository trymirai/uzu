use std::{path::PathBuf, sync::OnceLock};

static BUILD_DEBUG: OnceLock<bool> = OnceLock::new();
static BUILD_ALWAYS: OnceLock<bool> = OnceLock::new();
static BUILD_CLEAN: OnceLock<bool> = OnceLock::new();
static VARIANT_MANIFEST: OnceLock<Option<PathBuf>> = OnceLock::new();

fn env_flag(name: &str) -> bool {
    println!("cargo::rerun-if-env-changed={name}");
    std::env::var(name).is_ok()
}

pub fn build_debug() -> bool {
    *BUILD_DEBUG.get_or_init(|| env_flag("BUILD_DEBUG"))
}

pub fn build_always() -> bool {
    *BUILD_ALWAYS.get_or_init(|| env_flag("BUILD_ALWAYS"))
}

pub fn build_clean() -> bool {
    *BUILD_CLEAN.get_or_init(|| env_flag("BUILD_CLEAN"))
}

/// Extra destination for the instantiated-variant manifest, on top of the copy always
/// written into `OUT_DIR`. Set it to a stable path to diff manifests across builds.
pub fn variant_manifest() -> Option<&'static PathBuf> {
    VARIANT_MANIFEST
        .get_or_init(|| {
            println!("cargo::rerun-if-env-changed=UZU_VARIANT_MANIFEST");
            std::env::var_os("UZU_VARIANT_MANIFEST").map(PathBuf::from)
        })
        .as_ref()
}
