mod backend;
mod buffer;
mod capture;
mod command_buffer;
mod context;
mod decompression;
mod error;
mod gpu_types;
mod heaps;
mod kernel;
mod metal_extensions;
mod timestamp_recorder;

pub use backend::Metal;
#[cfg(test)]
pub use gpu_types::kv_layout::KVLayout;

/// Profiling aid: kernels named in `UZU_DUPLICATE_KERNELS` (comma-separated) run a second time after a barrier, so the
/// change in step time is what one more dependent dispatch of them costs. Results are not meaningful while it is set.
pub(crate) fn duplicate_dispatch(kernel_name: &str) -> bool {
    let count = DISPATCH_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    trace_dispatch(count, kernel_name);
    static NAMES: std::sync::OnceLock<Box<[Box<str>]>> = std::sync::OnceLock::new();
    let names = NAMES.get_or_init(|| {
        std::env::var("UZU_DUPLICATE_KERNELS")
            .map(|names| names.split(',').filter(|name| !name.is_empty()).map(Box::from).collect())
            .unwrap_or_default()
    });
    !names.is_empty() && names.iter().any(|name| &**name == kernel_name)
}

/// Profiling aid: generated kernel dispatches and dispatch barriers encoded so far.
pub(crate) static DISPATCH_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub(crate) static BARRIER_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub fn dispatch_counters() -> (u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (DISPATCH_COUNT.load(Relaxed), BARRIER_COUNT.load(Relaxed))
}

/// Profiling aid: with `UZU_TRACE_DISPATCHES=first,count`, prints dispatches `first..first + count` and the barriers
/// between them.
pub(crate) fn trace_dispatch(
    index: u64,
    label: &str,
) {
    static RANGE: std::sync::OnceLock<Option<(u64, u64)>> = std::sync::OnceLock::new();
    let range = RANGE.get_or_init(|| {
        let value = std::env::var("UZU_TRACE_DISPATCHES").ok()?;
        let (first, count) = value.split_once(',')?;
        Some((first.parse().ok()?, count.parse().ok()?))
    });
    if let Some((first, count)) = range
        && (*first..first + count).contains(&index)
    {
        eprintln!("dispatch {index}: {label}");
    }
}
