fn main() {
    let target_vendor = std::env::var("CARGO_CFG_TARGET_VENDOR").unwrap_or_default();
    if target_vendor == "apple" {
        println!("cargo:rustc-link-lib=framework=IOKit");
    }
}
