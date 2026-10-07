use crate::common::kernel::KernelBufferAccess;

#[derive(Debug, Clone)]
pub enum SlangArgumentType {
    Ptr(KernelBufferAccess),
    Constant(Box<str>),
    Specialize(Box<str>),
    Axis(Box<str>, Box<str>),
    Groups(Box<str>),
    Threads(Box<str>),
}
