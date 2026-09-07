#[cfg(not(target_family = "wasm"))]
use futures_util::future::BoxFuture as PlatformBoxFuture;
#[cfg(target_family = "wasm")]
use futures_util::future::LocalBoxFuture as PlatformBoxFuture;

pub type BoxFuture<'a, T> = PlatformBoxFuture<'a, T>;
