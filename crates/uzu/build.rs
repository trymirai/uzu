use std::env;

fn main() {
    #[cfg(feature = "bindings-napi")]
    napi_build::setup();
    #[cfg(feature = "bindings-pyo3")]
    pyo3_build_config::add_extension_module_link_args();

    let target_os = env::var("CARGO_CFG_TARGET_OS").expect("Failed to get CARGO_CFG_TARGET_OS");
    let metal_backend =
        cfg!(feature = "backend-metal") && matches!(target_os.as_ref(), "macos" | "ios" | "tvos" | "visionos");
    println!("cargo::rustc-check-cfg=cfg(metal_backend)");
    if metal_backend {
        println!("cargo::rustc-cfg=metal_backend");
    }
}
