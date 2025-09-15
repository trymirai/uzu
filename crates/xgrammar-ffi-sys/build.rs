use std::{env, path::PathBuf};

fn main() {
    // Optional fast path for CI/tests: build a stub with no XGrammar linkage
    if env::var("XGRAMMAR_STUB").ok().as_deref() == Some("1") {
        let mut build = cc::Build::new();
        build.cpp(true).flag("-std=c++17");
        build.include("src");
        build.file("src/xgrammar_stub.cc");
        build.compile("xgrshim");
        return;
    }

    // If a prebuilt XGrammar install is provided, use it.
    if let Ok(dir) = env::var("XGRAMMAR_DIR") {
        println!("cargo:rustc-link-search=native={}/lib", dir);
        println!("cargo:rustc-link-lib=static=xgrammar");
        println!("cargo:rustc-link-lib=c++");
        println!("cargo:include={}/include", dir);
    } else {
        // Build third_party/xgrammar via CMake
        let manifest_dir =
            PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
        let xg_src = manifest_dir
            .parent()
            .and_then(|p| p.parent())
            .unwrap()
            .join("third_party")
            .join("xgrammar");

        let dst = cmake::Config::new(&xg_src)
            .define("CMAKE_BUILD_TYPE", "Release")
            .define("CMAKE_OSX_ARCHITECTURES", "arm64")
            .build();

        println!("cargo:rustc-link-search=native={}", dst.display());
        println!("cargo:rustc-link-lib=static=xgrammar");
        println!("cargo:rustc-link-lib=c++");

        // Provide include hint for downstream build scripts if needed
        println!("cargo:include={}", xg_src.join("include").to_string_lossy());
    }

    // Build the C++ shim
    let mut build = cc::Build::new();
    build.cpp(true).flag("-std=c++17");

    // Include paths: allow both vendored header and xgrammar's dlpack
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let xg_inc = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .unwrap()
        .join("third_party")
        .join("xgrammar")
        .join("include");
    let dlpack_inc = manifest_dir.join("src").join("dlpack");
    build.include(&xg_inc);
    build.include(&dlpack_inc);

    build.file("src/xgrammar_ffi.cc");
    build.compile("xgrshim");
}
