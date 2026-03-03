//! Fast Walsh-Hadamard Transform encodable block.
//!
//! Four modes:
//! - `Full`: transforms the entire row (N must be power of 2 in [64, 8192])
//! - `SimdShuffleBlock`: block-wise using simd_shuffle_xor with threadgroup preloading
//! - `Block`: block-wise using the radix-16 Fwht kernel per block
//! - `Decomposed`: factors the dimension as m * 2^k (m in {12, 20, 28}), applies
//!    a power-of-2 Hadamard for the 2^k part then a dense O(m^2) codelet for the m part

use std::ops::Deref;

use super::{EncodableBlock, EncodingParameters};
use crate::{
    DataType,
    backends::common::{
        Backend,
        kernel::{FwhtKernel, FwhtMKernel, FwhtSimdBlockKernel, Kernels},
    },
    forward_pass::state::{ArrayId, ForwardPassState},
};

pub enum FwhtMode {
    Full,
    SimdShuffleBlock { block_size: u32 },
    Block { block_size: u32 },
    Decomposed { n: u32, m: u32 },
}

/// Factors `dim` as `m * 2^k` where m in {12, 20, 28} and 2^k in [32, 8192].
pub fn decompose_hadamard(dim: usize) -> Option<(u32, u32)> {
    for m in [12, 20, 28] {
        if dim % m == 0 {
            let n = dim / m;
            if n.is_power_of_two() && n >= 32 && n <= 8192 {
                return Some((n as u32, m as u32));
            }
        }
    }
    None
}

pub struct Fwht<B: Backend> {
    full_kernel: Option<<B::Kernels as Kernels>::FwhtKernel>,
    simd_block_kernel: Option<<B::Kernels as Kernels>::FwhtSimdBlockKernel>,
    m_kernel: Option<<B::Kernels as Kernels>::FwhtMKernel>,
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
        let (full_kernel, simd_block_kernel, m_kernel, scale) = match &mode {
            FwhtMode::Full => {
                let kernel = <B::Kernels as Kernels>::FwhtKernel::new(context, data_type, row_dimension as i32)?;
                let scale = 1.0f32 / (row_dimension as f32).sqrt();
                (Some(kernel), None, None, scale)
            },
            FwhtMode::SimdShuffleBlock { block_size } => {
                let kernel = <B::Kernels as Kernels>::FwhtSimdBlockKernel::new(context, data_type, *block_size as i32)?;
                let scale = 1.0f32 / (32f32).sqrt();
                (None, Some(kernel), None, scale)
            },
            FwhtMode::Block { block_size } => {
                let kernel = <B::Kernels as Kernels>::FwhtKernel::new(context, data_type, *block_size as i32)?;
                let scale = 1.0f32 / (*block_size as f32).sqrt();
                (Some(kernel), None, None, scale)
            },
            FwhtMode::Decomposed { n, m } => {
                let fwht_kernel = <B::Kernels as Kernels>::FwhtKernel::new(context, data_type, *n as i32)?;
                let m_kernel = <B::Kernels as Kernels>::FwhtMKernel::new(context, data_type, *m as i32)?;
                let scale = 1.0f32 / (row_dimension as f32).sqrt();
                (Some(fwht_kernel), None, Some(m_kernel), scale)
            },
        };

        Ok(Self {
            full_kernel,
            simd_block_kernel,
            m_kernel,
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
            FwhtMode::SimdShuffleBlock { block_size } => {
                let num_blocks = self.row_dimension as u32 / block_size;
                self.simd_block_kernel.as_ref().expect("FwhtSimdBlockKernel missing").encode(
                    buffer.deref(),
                    batch_size as u32 * num_blocks,
                    self.scale,
                    encoder,
                );
            },
            FwhtMode::Block { block_size } => {
                let num_blocks = self.row_dimension as u32 / block_size;
                self.full_kernel.as_ref().expect("FwhtKernel missing for Block mode").encode(
                    buffer.deref(),
                    batch_size as u32 * num_blocks,
                    self.scale,
                    encoder,
                );
            },
            FwhtMode::Decomposed { n, m } => {
                self.full_kernel.as_ref().expect("FwhtKernel missing for Decomposed mode").encode(
                    buffer.deref(),
                    batch_size as u32 * m,
                    1.0f32,
                    encoder,
                );
                self.m_kernel.as_ref().expect("FwhtMKernel missing for Decomposed mode").encode(
                    buffer.deref(),
                    batch_size as u32,
                    *n,
                    self.scale,
                    encoder,
                );
            },
        }
    }
}
