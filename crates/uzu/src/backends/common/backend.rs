use std::{error::Error, fmt::Debug};

use super::{CommandBuffer, Context, CopyEncoder, Event, Kernels, NativeBuffer};

pub trait Backend: Debug + Clone {
    type Context: Context<Backend = Self>;
    type NativeBuffer: NativeBuffer<Backend = Self>;
    type CommandBuffer: CommandBuffer<Backend = Self>;
    type ComputeEncoder;
    type CopyEncoder: CopyEncoder<Backend = Self>;
    type Event: Event<Backend = Self>;
    type Kernels: Kernels<Backend = Self>;
    type Error: Error + Debug;
}
