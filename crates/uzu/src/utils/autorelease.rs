#[cfg(feature = "metal")]
use objc2::rc::autoreleasepool;

/// Here we check the "metal" feature because Metal backend objects require an autorelease pool,
/// while other backends, such as CPU, do not.
pub fn maybe_with_autoreleasepool<T, F: FnOnce() -> T>(action: F) -> T {
    #[cfg(feature = "metal")]
    return autoreleasepool(|_pool| action());

    #[cfg(not(feature = "metal"))]
    action()
}
