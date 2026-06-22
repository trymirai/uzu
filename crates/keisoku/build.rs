fn main() {
    // IOReport private symbols are resolved at runtime via `kanka` (dlsym), so
    // only the public IOKit + CoreFoundation symbols (IOService*/IORegistry*, CF*)
    // are linked here. macOS only — iOS sandboxes IOReport/IORegistry telemetry.
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "macos" {
        println!("cargo:rustc-link-lib=framework=IOKit");
        println!("cargo:rustc-link-lib=framework=CoreFoundation");
    }
}
