use std::{
    pin::Pin,
    task::{Context, Poll},
};

use wasm_bindgen::JsValue;
use web_sys::js_sys::Promise;

pub struct JsFuture(wasm_bindgen_futures::JsFuture);

impl From<Promise> for JsFuture {
    fn from(promise: Promise) -> Self {
        Self(promise.into())
    }
}

impl Future for JsFuture {
    type Output = Result<JsValue, JsValue>;

    fn poll(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Self::Output> {
        Pin::new(&mut self.get_mut().0).poll(cx)
    }
}

#[cfg(not(target_feature = "atomics"))]
unsafe impl Send for JsFuture {}
