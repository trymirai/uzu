use crate::backends::common::{Backend, CommandBuffer};

pub trait EncodableBlock<B: Backend, Input>: Send + Sync {
    type Kernel;
    type Output;
    type Error;

    fn encode(
        &self,
        input: Input,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<Self::Output, Self::Error>;
}
