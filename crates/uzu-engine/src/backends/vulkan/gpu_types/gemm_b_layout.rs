use derive_more::Display;

#[repr(C)]
#[derive(Debug, Display, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GemmBLayout {
    #[default]
    RowMajor,
    Tiled,
}
