#[derive(PartialEq, Debug)]
pub enum ConstantType {
    Scalar,
    Array,
    Struct,
}

#[derive(PartialEq, Debug)]
pub enum KernelArgumentType {
    Buffer,
    Constant((Box<str>, ConstantType)),
}

#[derive(PartialEq, Debug)]
pub struct KernelArgument {
    pub name: Box<str>,
    pub ty: KernelArgumentType,
}

#[derive(PartialEq, Debug)]
pub enum KernelParameterType {
    DType,
    Specialization(Box<str>),
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

#[derive(PartialEq, Debug, Clone)]
pub struct StructField {
    pub name: Box<str>,
    pub ty: Box<str>,
}

#[derive(PartialEq, Debug, Clone)]
pub struct Struct {
    pub name: Box<str>,
    pub fields: Box<[StructField]>,
}
