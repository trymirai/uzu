use std::{ops::Range, time::Duration};

use super::Backend;

pub trait CommandBuffer {
    type Backend: Backend<CommandBuffer = Self>;

    type Initial: CommandBufferInitial<CommandBuffer = Self>;
    type Encoding: CommandBufferEncoding<CommandBuffer = Self>;
    type Executable: CommandBufferExecutable<CommandBuffer = Self>;
    type Pending: CommandBufferPending<CommandBuffer = Self>;
    type Completed: CommandBufferCompleted<CommandBuffer = Self>;
}

pub trait CommandBufferInitial {
    type CommandBuffer: CommandBuffer<Initial = Self>;

    fn start_encoding(self) -> <Self::CommandBuffer as CommandBuffer>::Encoding;
}

pub trait CommandBufferEncoding {
    type CommandBuffer: CommandBuffer<Encoding = Self>;

    fn encode_copy(
        &mut self,
        src: &<<Self::CommandBuffer as CommandBuffer>::Backend as Backend>::Buffer,
        dst: &mut <<Self::CommandBuffer as CommandBuffer>::Backend as Backend>::Buffer,
        size: usize,
    );

    fn encode_copy_ranges(
        &mut self,
        src: (&<<Self::CommandBuffer as CommandBuffer>::Backend as Backend>::Buffer, usize),
        dst: (&<<Self::CommandBuffer as CommandBuffer>::Backend as Backend>::Buffer, usize),
        size: usize,
    );

    fn encode_fill(
        &mut self,
        dst: &mut <<Self::CommandBuffer as CommandBuffer>::Backend as Backend>::Buffer,
        range: Range<usize>,
        value: u8,
    );

    fn encode_wait_for_event(
        &mut self,
        event: &<<Self::CommandBuffer as CommandBuffer>::Backend as Backend>::Event,
        value: u64,
    );

    fn encode_signal_event(
        &mut self,
        event: &<<Self::CommandBuffer as CommandBuffer>::Backend as Backend>::Event,
        value: u64,
    );

    fn add_completion_handler(
        &mut self,
        handler: impl FnOnce(
            Result<
                &<Self::CommandBuffer as CommandBuffer>::Completed,
                <<Self::CommandBuffer as CommandBuffer>::Backend as Backend>::Error,
            >,
        ) + 'static,
    );

    fn end_encoding(self) -> <Self::CommandBuffer as CommandBuffer>::Executable;
}

pub trait CommandBufferExecutable {
    type CommandBuffer: CommandBuffer<Executable = Self>;

    fn submit(self) -> <Self::CommandBuffer as CommandBuffer>::Pending;
}

pub trait CommandBufferPending {
    type CommandBuffer: CommandBuffer<Pending = Self>;

    fn is_completed(&self) -> bool;

    fn wait_until_completed(
        self
    ) -> Result<
        <Self::CommandBuffer as CommandBuffer>::Completed,
        <<Self::CommandBuffer as CommandBuffer>::Backend as Backend>::Error,
    >;
}

pub trait CommandBufferCompleted {
    type CommandBuffer: CommandBuffer<Completed = Self>;

    fn is_completed(&self) -> bool;

    fn gpu_execution_time(&self) -> Option<Duration>;
}
