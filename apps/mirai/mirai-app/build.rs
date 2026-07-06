fn main() {
    let key = "MIRAI_BUNDLED_API_KEY";
    println!("cargo:rerun-if-env-changed={key}");
    if let Ok(value) = std::env::var(key) {
        println!("cargo:rustc-env={key}={value}");
    }
}
