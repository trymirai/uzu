use std::time::Duration;

use crate::backends::common::{Backend, Buffer, BufferRangeMut, BufferRangeRef};

pub trait CommandBuffer {
    type Backend: Backend<CommandBuffer = Self>;

    type Initial: CommandBufferInitial<CommandBuffer = Self>;
    type Encoding: CommandBufferEncoding<CommandBuffer = Self>;
    type Executable: CommandBufferExecutable<CommandBuffer = Self>;
    type Pending: CommandBufferPending<CommandBuffer = Self>;
    type Completed: CommandBufferCompleted<CommandBuffer = Self>;
}

pub trait CommandBufferInitial: Send {
    type CommandBuffer: CommandBuffer<Initial = Self>;

    fn start_encoding(self) -> <Self::CommandBuffer as CommandBuffer>::Encoding;
}

pub trait CommandBufferEncoding {
    type CommandBuffer: CommandBuffer<Encoding = Self>;

    fn encode_copy<
        Src: Buffer<Backend = <Self::CommandBuffer as CommandBuffer>::Backend>,
        Dst: Buffer<Backend = <Self::CommandBuffer as CommandBuffer>::Backend>,
    >(
        &mut self,
        src: BufferRangeRef<Src>,
        dst: BufferRangeMut<Dst>,
    );

    fn encode_fill<Dst: Buffer<Backend = <Self::CommandBuffer as CommandBuffer>::Backend>>(
        &mut self,
        dst: BufferRangeMut<Dst>,
        value: u8,
    );

    fn push_debug_group(
        &mut self,
        name: &str,
    );

    fn pop_debug_group(&mut self);

    fn end_encoding(self) -> <Self::CommandBuffer as CommandBuffer>::Executable;
}

pub trait CommandBufferExecutable: Send {
    type CommandBuffer: CommandBuffer<Executable = Self>;

    fn submit(self) -> <Self::CommandBuffer as CommandBuffer>::Pending;
}

pub trait CommandBufferPending: Send {
    type CommandBuffer: CommandBuffer<Pending = Self>;

    fn wait_until_completed(
        self
    ) -> Result<
        <Self::CommandBuffer as CommandBuffer>::Completed,
        <<Self::CommandBuffer as CommandBuffer>::Backend as Backend>::Error,
    >;
}

pub trait CommandBufferCompleted: Send {
    type CommandBuffer: CommandBuffer<Completed = Self>;

    fn gpu_execution_time(&self) -> Duration;
}
