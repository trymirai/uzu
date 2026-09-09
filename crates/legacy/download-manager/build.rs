fn main() {
    #[cfg(feature = "bindings-napi")]
    napi_build::setup();
    #[cfg(feature = "bindings-pyo3")]
    pyo3_build_config::add_extension_module_link_args();
}
