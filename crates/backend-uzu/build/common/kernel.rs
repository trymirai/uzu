use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, PartialEq, Debug, Clone, Copy)]
pub enum KernelBufferAccess {
    Read,
    ReadWrite,
}

#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
pub enum KernelArgumentType {
    Buffer(KernelBufferAccess),
    Constant(Box<str>),
}

#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
pub struct KernelArgument {
    pub name: Box<str>,
    pub conditional: bool,
    pub ty: KernelArgumentType,
}

#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
pub enum KernelParameterType {
    Type,
    Value(Box<str>),
}

#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
pub struct KernelParameter {
    pub name: Box<str>,
    pub ty: KernelParameterType,
}

#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
pub struct Kernel {
    pub name: Box<str>,
    pub parameters: Box<[KernelParameter]>,
    pub arguments: Box<[KernelArgument]>,
}
