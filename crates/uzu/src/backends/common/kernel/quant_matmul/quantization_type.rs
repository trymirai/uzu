use crate::backends::common::Backend;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizedMatmulType {
    ZeroPoint,
    Mlx,
}

impl QuantizedMatmulType {
    pub fn split_buffers<'a, B: Backend>(
        &self,
        buffer: &'a B::Buffer,
    ) -> (Option<&'a B::Buffer>, Option<&'a B::Buffer>) {
        match self {
            Self::ZeroPoint => (Some(buffer), None),
            Self::Mlx => (None, Some(buffer)),
        }
    }
}
