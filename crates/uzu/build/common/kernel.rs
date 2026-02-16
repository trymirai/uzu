#[derive(PartialEq, Debug)]
pub enum KernelArgumentType {
    Buffer,
    Constant(Box<str>),
    Scalar(Box<str>),
}

#[derive(PartialEq, Debug)]
pub struct KernelArgument {
    pub name: Box<str>,
    pub conditional: bool,
    pub ty: KernelArgumentType,
}

#[derive(PartialEq, Debug)]
pub enum KernelParameterType {
    Type,
    Value(Box<str>),
}

#[derive(PartialEq, Debug)]
pub struct KernelParameter {
    pub name: Box<str>,
    pub ty: KernelParameterType,
}

#[derive(PartialEq, Debug)]
pub struct Kernel {
    pub name: Box<str>,
    pub parameters: Box<[KernelParameter]>,
    pub arguments: Box<[KernelArgument]>,
}
