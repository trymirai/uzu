pub mod common;

pub mod cpu;

#[cfg(metal_backend)]
pub mod metal;

#[cfg(webgpu_backend)]
pub mod webgpu;

macro_rules! select_backend {
    ($expr:expr, $unk:expr) => {{
        let default = if cfg!(metal_backend) {
            "metal"
        } else if cfg!(webgpu_backend) {
            "webgpu"
        } else {
            "cpu"
        };

        match std::env::var("UZU_BACKEND").map(|s| s.to_lowercase()).as_deref().unwrap_or(default) {
            "cpu" => {
                type B = crate::backends::cpu::Cpu;
                $expr
            },
            #[cfg(metal_backend)]
            "metal" => {
                type B = crate::backends::metal::Metal;
                $expr
            },
            #[cfg(webgpu_backend)]
            "webgpu" => {
                type B = crate::backends::webgpu::WebGPU;
                $expr
            },
            _ => Err($unk),
        }
    }};
}
pub(crate) use select_backend;
