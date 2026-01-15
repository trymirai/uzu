use std::sync::OnceLock;
use std::time::Instant;

static START: OnceLock<Instant> = OnceLock::new();

pub fn log(args: std::fmt::Arguments) {
    let start = START.get_or_init(Instant::now);
    let elapsed_ms = start.elapsed().as_millis();
    println!("cargo::warning=(build-debug) [{elapsed_ms}ms] {args}");
}

#[macro_export]
macro_rules! debug_log {
    ($($arg:tt)*) => {{
        if cfg!(feature = "build-debug") {
            $crate::common::build_debug::log(format_args!($($arg)*));
        }
    }};
}
