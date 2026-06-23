pub const UZU_CAPTURE_BENCH: &str = "UZU_CAPTURE_BENCH";
pub const UZU_CAPTURE_BENCH_FILTER: &str = "UZU_CAPTURE_BENCH_FILTER";
pub const UZU_CAPTURE_BENCH_DIR: &str = "UZU_CAPTURE_BENCH_DIR";
pub const METAL_CAPTURE_ENABLED: &str = "METAL_CAPTURE_ENABLED";

pub fn enabled(name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("yes") || v.eq_ignore_ascii_case("true"))
}
