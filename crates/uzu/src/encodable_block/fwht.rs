//! Fast Walsh-Hadamard Transform encodable block.
//!
//! Two modes:
//! - `Full`: transforms the entire row (N must be power of 2 and match a compiled variant)
//! - `Block`: divides the row into chunks of `block_size` and transforms each independently

use std::ops::Deref;

use super::{EncodableBlock, EncodingParameters};
use crate::{
    DataType,
    backends::common::{
        Backend,
        kernel::{FwhtBlockKernel, FwhtKernel, Kernels},
    },
    forward_pass::state::{ArrayId, ForwardPassState},
};

pub enum FwhtMode {
    Full,
    Block { block_size: u32 },
}

pub struct Fwht<B: Backend> {
    full_kernel: Option<<B::Kernels as Kernels>::FwhtKernel>,
    block_kernel: Option<<B::Kernels as Kernels>::FwhtBlockKernel>,
    mode: FwhtMode,
    array_id: ArrayId,
    row_dimension: usize,
    scale: f32,
}

impl<B: Backend> Fwht<B> {
    pub fn new(
        context: &B::Context,
        data_type: DataType,
        mode: FwhtMode,
        array_id: ArrayId,
        row_dimension: usize,
    ) -> Result<Self, B::Error> {
        let (full_kernel, block_kernel, scale) = match &mode {
            FwhtMode::Full => {
                let kernel = <B::Kernels as Kernels>::FwhtKernel::new(context, data_type, row_dimension as i32)?;
                let scale = 1.0f32 / (row_dimension as f32).sqrt();
                (Some(kernel), None, scale)
            },
            FwhtMode::Block {
                block_size,
            } => {
                let kernel = <B::Kernels as Kernels>::FwhtBlockKernel::new(context, data_type, *block_size as i32)?;
                let scale = 1.0f32 / (*block_size as f32).sqrt();
                (None, Some(kernel), scale)
            },
        };

        Ok(Self {
            full_kernel,
            block_kernel,
            mode,
            array_id,
            row_dimension,
            scale,
        })
    }
}

impl<B: Backend> EncodableBlock<B> for Fwht<B> {
    fn supports_shared_encoder(&self) -> bool {
        true
    }

    fn encode_with_shared_encoder(
        &self,
        state: &mut ForwardPassState<B>,
        _parameters: &EncodingParameters<B>,
        encoder: &mut B::ComputeEncoder,
    ) {
        let arrays = state.arrays(&[self.array_id]);
        let array = arrays[0].borrow_mut();
        let batch_size = state.active_suffix_length();
        let buffer_rc = array.buffer();
        let buffer = buffer_rc.borrow();

        match &self.mode {
            FwhtMode::Full => {
                self.full_kernel.as_ref().expect("FwhtKernel missing for Full mode").encode(
                    buffer.deref(),
                    batch_size as u32,
                    self.scale,
                    encoder,
                );
            },
            FwhtMode::Block {
                ..
            } => {
                self.block_kernel.as_ref().expect("FwhtBlockKernel missing for Block mode").encode(
                    buffer.deref(),
                    batch_size as u32,
                    self.row_dimension as u32,
                    self.scale,
                    encoder,
                );
            },
        }
    }
}
