fn main() {
    #[cfg(feature = "bindings-napi")]
    napi_build::setup();
}
