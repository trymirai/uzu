use std::{sync::OnceLock, time::Instant};

use crate::common::envs;

static START: OnceLock<Instant> = OnceLock::new();

pub fn elapsed_ms() -> u128 {
    START.get_or_init(Instant::now).elapsed().as_millis()
}

pub fn _debug_log(args: std::fmt::Arguments) {
    if envs::build_debug() {
        println!("cargo::warning=(build-debug) [{}ms] {}", elapsed_ms(), args);
    }
}

#[macro_export]
macro_rules! debug_log {
    ($($arg:tt)*) => {{
        $crate::common::logging::_debug_log(format_args!($($arg)*));
    }};
}
