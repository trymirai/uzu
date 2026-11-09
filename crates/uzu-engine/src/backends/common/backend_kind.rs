#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendKind {
    Cpu,
    Metal,
    WebGpu,
}

impl BackendKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Metal => "metal",
            Self::WebGpu => "webgpu",
        }
    }
}
