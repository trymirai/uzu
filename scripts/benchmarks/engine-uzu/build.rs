fn main() {
    // The memory counters read mach task info, which only Apple platforms have.
    if std::env::var("CARGO_CFG_TARGET_VENDOR").as_deref() != Ok("apple") {
        return;
    }
    let header = "../common-cpp/src/memory_counters.h";
    let source = "../common-cpp/src/memory_counters.c";
    println!("cargo:rerun-if-changed={source}");
    println!("cargo:rerun-if-changed={header}");

    cc::Build::new().file(source).include("../common-cpp/src").compile("benchmark_memory_counters");
}
