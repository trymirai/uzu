use shader_slang::ScalarType;

#[derive(Debug, PartialEq)]
pub struct SlangStructFieldAbi {
    pub name: String,
    pub offset: usize,
    pub size: usize,
    pub scalar: ScalarType,
    pub array: Option<(usize, usize)>,
}
