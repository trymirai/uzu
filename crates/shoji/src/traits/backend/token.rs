use crate::traits::backend::Backend as BackendTrait;

pub type Input = Vec<u64>;
pub type Output = u64;

pub trait Backend: BackendTrait<StreamInput = Input, StreamOutput = Output> {}

impl<T> Backend for T where T: BackendTrait<StreamInput = Input, StreamOutput = Output> {}
