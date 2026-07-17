use std::time::Duration;

use super::{Backend, Buffer, BufferRangeMut, BufferRangeRef};

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

#[derive(Debug, Clone, PartialEq)]
pub struct AccessFlags {
    pub compute_read: bool,
    pub compute_write: bool,
    pub copy_read: bool,
    pub copy_write: bool,
}

impl AccessFlags {
    pub fn empty() -> Self {
        Self {
            compute_read: false,
            compute_write: false,
            copy_read: false,
            copy_write: false,
        }
    }

    pub fn with_compute_read(mut self) -> Self {
        self.compute_read = true;
        self
    }
    pub fn with_compute_write(mut self) -> Self {
        self.compute_write = true;
        self
    }
    pub fn with_copy_read(mut self) -> Self {
        self.copy_read = true;
        self
    }
    pub fn with_copy_write(mut self) -> Self {
        self.copy_write = true;
        self
    }

    pub fn compute_read() -> Self {
        Self::empty().with_compute_read()
    }
    pub fn compute_write() -> Self {
        Self::empty().with_compute_write()
    }
    pub fn copy_read() -> Self {
        Self::empty().with_copy_read()
    }
    pub fn copy_write() -> Self {
        Self::empty().with_copy_write()
    }
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

    fn encode_barrier(
        &mut self,
        after: AccessFlags,
        before: AccessFlags,
    );

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
