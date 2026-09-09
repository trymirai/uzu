#[cfg(not(target_family = "wasm"))]
use futures_util::stream::BoxStream as PlatformBoxStream;
#[cfg(target_family = "wasm")]
use futures_util::stream::LocalBoxStream as PlatformBoxStream;

pub type BoxStream<'a, T> = PlatformBoxStream<'a, T>;
