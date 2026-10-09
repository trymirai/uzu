use std::{mem::size_of_val, time::Duration};

use bytemuck::{AnyBitPattern, NoUninit};

use crate::{
    array::size_for_shape,
    backends::common::{Backend, BufferMut, BufferRef},
    data_type::DataType,
};

pub trait CommandBuffer {
    type Backend: Backend<CommandBuffer = Self>;

    type Encoding: CommandBufferEncoding<CommandBuffer = Self>;
    type Executable: CommandBufferExecutable<CommandBuffer = Self>;
    type Pending: CommandBufferPending<CommandBuffer = Self>;
    type Completed: CommandBufferCompleted<CommandBuffer = Self>;
}

pub trait CommandBufferEncoding {
    type CommandBuffer: CommandBuffer<Encoding = Self>;

    fn context(&self) -> &<<Self::CommandBuffer as CommandBuffer>::Backend as Backend>::Context;

    fn allocate_constant(
        &mut self,
        size: usize,
    ) -> Result<
        <<Self::CommandBuffer as CommandBuffer>::Backend as Backend>::ConstantBuffer,
        <<Self::CommandBuffer as CommandBuffer>::Backend as Backend>::Error,
    >;

    fn allocate_constant_from_slice(
        &mut self,
        data: &[impl NoUninit + AnyBitPattern],
    ) -> Result<
        <<Self::CommandBuffer as CommandBuffer>::Backend as Backend>::ConstantBuffer,
        <<Self::CommandBuffer as CommandBuffer>::Backend as Backend>::Error,
    > {
        let mut buffer = self.allocate_constant(size_of_val(data))?;
        buffer.copyin(data);
        Ok(buffer)
    }

    fn allocate_scratch(
        &mut self,
        size: usize,
    ) -> Result<
        <<Self::CommandBuffer as CommandBuffer>::Backend as Backend>::ScratchBuffer,
        <<Self::CommandBuffer as CommandBuffer>::Backend as Backend>::Error,
    >;

    fn allocate_scratch_for_shape(
        &mut self,
        shape: &[u32],
        data_type: DataType,
    ) -> Result<
        <<Self::CommandBuffer as CommandBuffer>::Backend as Backend>::ScratchBuffer,
        <<Self::CommandBuffer as CommandBuffer>::Backend as Backend>::Error,
    > {
        self.allocate_scratch(size_for_shape(shape, data_type))
    }

    fn encode_copy(
        &mut self,
        src: impl BufferRef<Backend = <Self::CommandBuffer as CommandBuffer>::Backend>,
        dst: impl BufferMut<Backend = <Self::CommandBuffer as CommandBuffer>::Backend>,
    );

    fn encode_fill(
        &mut self,
        dst: impl BufferMut<Backend = <Self::CommandBuffer as CommandBuffer>::Backend>,
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
