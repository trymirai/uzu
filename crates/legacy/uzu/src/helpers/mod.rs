#[cfg(not(target_family = "wasm"))]
mod reachability;
mod same_origin;
mod shared_access;

#[cfg(not(target_family = "wasm"))]
pub use reachability::is_endpoint_reachable;
pub use same_origin::same_origin;
pub use shared_access::SharedAccess;
