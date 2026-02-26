pub mod common;

pub mod cpu;
#[cfg(feature = "metal")]
pub mod metal;

macro_rules! select_backend {
    ($expr:expr, $unk:expr) => {{
        match std::env::var("UZU_BACKEND").map(|s| s.to_lowercase()).as_deref().unwrap_or("metal") {
            #[cfg(feature = "metal")]
            "metal" => {
                type B = crate::backends::metal::Metal;
                $expr
            },
            _ => Err($unk),
        }
    }};
}
pub(crate) use select_backend;
