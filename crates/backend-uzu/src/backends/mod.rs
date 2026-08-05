pub mod common;

#[cfg(backend = "cpu")]
pub mod cpu;
#[cfg(backend = "metal")]
pub mod metal;

macro_rules! select_backend {
    ($expr:expr, $unk:expr) => {{
        let default = if cfg!(backend = "metal") {
            "metal"
        } else if cfg!(backend = "cpu") {
            "cpu"
        } else {
            unreachable!()
        };

        // TODO: remove magic env var
        match std::env::var("UZU_BACKEND").map(|s| s.to_lowercase()).as_deref().unwrap_or(default) {
            #[cfg(backend = "cpu")]
            "cpu" => {
                type B = crate::backends::cpu::Cpu;
                $expr
            },
            #[cfg(backend = "metal")]
            "metal" => {
                type B = crate::backends::metal::Metal;
                $expr
            },
            _ => Err($unk),
        }
    }};
}
pub(crate) use select_backend;
