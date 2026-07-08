#![allow(dead_code)]

pub mod fs;
pub mod maybe;
pub mod process;
pub mod rt;
pub mod time;

#[cfg(target_family = "wasm")]
#[doc(hidden)]
pub mod __console {
    pub fn log(message: &str) {
        web_sys::console::log_1(&message.into());
    }

    pub fn error(message: &str) {
        web_sys::console::error_1(&message.into());
    }
}

#[macro_export]
macro_rules! printf {
    ($($arg:tt)*) => {
		#[cfg(not(target_family = "wasm"))]
        println!($($arg)*);

        #[cfg(target_family = "wasm")]
		{
			let string = format!($($arg)*);
			$crate::__console::log(&string);
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
			$crate::__console::error(&string);
		}
    };
}
