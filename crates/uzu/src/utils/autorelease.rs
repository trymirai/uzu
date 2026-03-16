#[cfg(feature = "metal")]
use objc2::rc::autoreleasepool;

/// Here checked metal feature because autoreleasepool is required for Metal backend objects
/// and another backends like CPU don't require autoreleasepool.
pub fn maybe_with_autorelease_pool<T, F: FnOnce() -> T>(action: F) -> T {
    #[cfg(feature = "metal")]
    return autoreleasepool(|_pool| action());

    #[cfg(not(feature = "metal"))]
    action()
}
