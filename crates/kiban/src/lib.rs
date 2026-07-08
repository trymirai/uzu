#![allow(dead_code)]

pub mod fs;
pub mod maybe;
pub mod process;
pub mod rt;
pub mod time;

#[macro_export]
macro_rules! printf {
    ($($arg:tt)*) => {
		#[cfg(not(target_family = "wasm"))]
        println!($($arg)*);

        #[cfg(target_family = "wasm")]
		{
			let string = format!($($arg)*);
			web_sys::console::log_1(&string.into());
		}
    };
}

#[macro_export]
macro_rules! eprintf {
    ($($arg:tt)*) => {
		#[cfg(not(target_family = "wasm"))]
        eprintln!($($arg)*);

        #[cfg(target_family = "wasm")]
		{
			let string = format!($($arg)*);
			web_sys::console::error_1(&string.into());
		}
    };
}
