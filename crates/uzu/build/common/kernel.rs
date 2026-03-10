#[derive(PartialEq, Debug, Clone)]
pub enum KernelBufferAccess {
    Read,
    ReadWrite,
}

#[derive(PartialEq, Debug, Clone)]
pub enum KernelArgumentType {
    Buffer(KernelBufferAccess),
    Constant(Box<str>),
    Scalar(Box<str>),
}

#[derive(PartialEq, Debug, Clone)]
pub struct KernelArgument {
    pub name: Box<str>,
    pub conditional: bool,
    pub ty: KernelArgumentType,
}

#[derive(PartialEq, Debug, Clone)]
pub enum KernelParameterType {
    Type,
    Value(Box<str>),
}

#[derive(PartialEq, Debug, Clone)]
pub struct KernelParameter {
    pub name: Box<str>,
    pub ty: KernelParameterType,
}

#[derive(PartialEq, Debug, Clone)]
pub struct Kernel {
    pub name: Box<str>,
    pub parameters: Box<[KernelParameter]>,
    pub arguments: Box<[KernelArgument]>,
}
