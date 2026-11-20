use std::{env, path::PathBuf};

use bindgen::Builder;
use cc::Build;

fn main() {
    if env::var("CARGO_CFG_TARGET_VENDOR").as_deref() != Ok("apple") {
        return;
    }
    let mut build = Build::new();
    let mut bindings = Builder::default()
        .allowlist_function("get_memory_counters|memory_counters_error_string|await_cooldown|get_gpu_energy")
        .derive_default(true)
        .layout_tests(false)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()));
    for name in ["memory_counters", "apple_temp_sensors", "energy_counters"] {
        let source = format!("../common-cpp/src/{name}.c");
        println!("cargo:rerun-if-changed={source}");
        bindings = bindings.header(format!("../common-cpp/src/{name}.h"));
        build.file(source);
    }
    bindings
        .generate()
        .expect("generate benchmark system bindings")
        .write_to_file(PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("system.rs"))
        .expect("write benchmark system bindings");
    build.include("../common-cpp/src").compile("benchmark_system");
    println!("cargo:rustc-link-lib=framework=IOKit");
    println!("cargo:rustc-link-lib=framework=CoreFoundation");
    println!("cargo:rustc-link-lib=IOReport");
}
