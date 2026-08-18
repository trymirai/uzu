use std::sync::OnceLock;

static BUILD_DEBUG: OnceLock<bool> = OnceLock::new();
static BUILD_ALWAYS: OnceLock<bool> = OnceLock::new();
static BUILD_CLEAN: OnceLock<bool> = OnceLock::new();

fn env_flag(name: &str) -> bool {
    println!("cargo::rerun-if-env-changed={name}");
    match std::env::var(name) {
        Ok(value) => matches!(value.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"),
        Err(_) => false,
    }
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
