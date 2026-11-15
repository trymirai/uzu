use std::{
    fmt,
    sync::{
        LazyLock,
        atomic::{AtomicU8, Ordering},
    },
};

/// 0 = follow `UZU_VERBOSE`, 1 = forced off, 2 = forced on.
static VERBOSE_OVERRIDE: AtomicU8 = AtomicU8::new(0);
static ENV_VERBOSE: LazyLock<bool> =
    LazyLock::new(|| std::env::var("UZU_VERBOSE").is_ok_and(|value| !matches!(value.as_str(), "" | "0" | "false")));

/// Whether diagnostic engine output (prefill/vision breakdowns, GPU operator
/// spans) should be printed. Enabled by `UZU_VERBOSE=1` in the environment or
/// programmatically via [`set_verbose`].
pub fn is_verbose() -> bool {
    match VERBOSE_OVERRIDE.load(Ordering::Relaxed) {
        1 => false,
        2 => true,
        _ => *ENV_VERBOSE,
    }
}

/// Force diagnostic engine output on or off, overriding the environment.
pub fn set_verbose(verbose: bool) {
    VERBOSE_OVERRIDE.store(if verbose { 2 } else { 1 }, Ordering::Relaxed);
}

/// Print to stderr only when verbose diagnostics are enabled.
pub fn log(args: fmt::Arguments<'_>) {
    if is_verbose() {
        eprintln!("{args}");
    }
}

/// Verbose-gated stderr logging for engine internals.
///
/// ```
/// use uzu_engine::engine_verbose;
/// engine_verbose!("🔍 [Prefill Breakdown] gpu = {:.3}s", 0.123);
/// ```
#[macro_export]
macro_rules! engine_verbose {
    ($($arg:tt)*) => {
        $crate::verbose::log(format_args!($($arg)*))
    };
}
