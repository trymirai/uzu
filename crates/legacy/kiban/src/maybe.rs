#![allow(dead_code)]

#[cfg(not(target_family = "wasm"))]
pub trait MaybeSend: std::marker::Send {}
#[cfg(not(target_family = "wasm"))]
impl<T: std::marker::Send + ?Sized> MaybeSend for T {}

#[cfg(target_family = "wasm")]
pub trait MaybeSend {}
#[cfg(target_family = "wasm")]
impl<T: ?Sized> MaybeSend for T {}

#[cfg(not(target_family = "wasm"))]
pub trait MaybeSync: std::marker::Sync {}
#[cfg(not(target_family = "wasm"))]
impl<T: std::marker::Sync + ?Sized> MaybeSync for T {}

#[cfg(target_family = "wasm")]
pub trait MaybeSync {}
#[cfg(target_family = "wasm")]
impl<T: ?Sized> MaybeSync for T {}
