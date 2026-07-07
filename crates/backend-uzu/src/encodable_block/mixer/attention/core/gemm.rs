use std::cell::RefCell;

use crate::{
    array::size_for_shape,
    backends::common::{
        Allocation, Backend, BufferArg, Encoder, Kernels,
        gpu_types::AttnParams,
        kernel::attention_gemm::{AttentionGemmArgs, AttentionGemmDispatch},
    },
    data_type::DataType,
    encodable_block::mixer::attention::core::{AttentionCoreEncodeArguments, AttentionCoreNewArguments},
};

pub struct AttentionGemmCore<B: Backend> {
    head_dim: usize,
    num_groups: usize,
    num_q_heads: usize,
    sliding_window_size: Option<usize>,
    scale: Option<f32>,
    data_type: DataType,
    kernel: RefCell<<B::Kernels as Kernels>::AttentionGemmDispatch>,
}

impl<B: Backend> AttentionGemmCore<B> {
    pub fn new(
        arguments: &AttentionCoreNewArguments,
        _use_mxu: bool,
        context: &B::Context,
    ) -> Result<Self, B::Error> {
        let bk = if arguments.head_dim < 128 {
            32
        } else {
            16
        };
        let kernel = <B::Kernels as Kernels>::AttentionGemmDispatch::new(
            context,
            arguments.data_type,
            bk as u32,
            arguments.head_dim as u32,
            arguments.is_kv_cache_ring,
            arguments.is_causal,
            arguments.is_trie,
            arguments.sliding_window_size.is_some(),
            arguments.has_sinks,
        )?;

        Ok(Self {
            head_dim: arguments.head_dim,
            num_groups: arguments.num_groups,
            num_q_heads: arguments.num_q_heads,
            sliding_window_size: arguments.sliding_window_size,
            scale: arguments.scale,
            data_type: arguments.data_type,
            kernel: RefCell::new(kernel),
        })
    }

    pub fn encode<'a, KT: BufferArg<'a, B>, VT: BufferArg<'a, B>>(
        &self,
        arguments: AttentionCoreEncodeArguments<'a, B, KT, VT>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let mut output = encoder.allocate_constant(size_for_shape(
            &[arguments.suffix_length, self.num_q_heads, self.head_dim],
            self.data_type,
        ))?;
        self.kernel.borrow_mut().encode(
            AttentionGemmArgs {
                q: arguments.queries,
                k: arguments.keys,
                v: arguments.values,
                o: &mut output,
                params: AttnParams {
                    q_strides: [0, (arguments.suffix_length * self.head_dim) as u64, self.head_dim as u64],
                    k_strides: [0, self.head_dim as u64, (self.num_groups * self.head_dim) as u64],
                    v_strides: [0, self.head_dim as u64, (self.num_groups * self.head_dim) as u64],
                    o_strides: [0, self.head_dim as u64, (self.num_q_heads * self.head_dim) as u64],
                    gqa_factor: (self.num_q_heads / self.num_groups) as u32,
                    scale: self.scale.unwrap_or(1.0f32 / (self.head_dim as f32).sqrt()),
                    q_len: arguments.suffix_length as u32,
                    k_len: (arguments.state_type.physical_prefix_length() + arguments.suffix_length) as u32,
                    q_off: arguments.state_type.physical_prefix_length() as u32,
                    nq_aligned: 0,
                    q_rem: 0,
                    nk: 0,
                    nk_aligned: 0,
                    k_rem: 0,
                },
                ring_params: arguments.state_type.ring_params(),
                trie: arguments.trie,
                sliding_window_size: self.sliding_window_size.map(|sliding_window_size| sliding_window_size as u32),
                sinks: arguments.sinks,
                num_heads: self.num_q_heads as u32,
                suffix_length: arguments.suffix_length as u32,
            },
            encoder,
        )?;

        Ok(output)
    }
}

#[cfg(test)]
#[path = "../../../../../unit/encodable_block/attention_gemm_test.rs"]
mod tests;
