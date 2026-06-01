use std::mem::MaybeUninit;

use bitflags::bitflags;

use crate::{
    array::size_for_shape,
    backends::common::{
        Allocation, Backend, Encoder, Kernels,
        gpu_types::AttnParams,
        kernel::{AttentionGemmKernel, BufferArg},
    },
    data_type::DataType,
    encodable_block::mixer::attention::core::{AttentionCoreEncodeArguments, AttentionCoreNewArguments},
};

const BQ: usize = 32;

bitflags! {
    pub struct AttentionGemmKey: usize {
        const ALIGNED_Q = 1 << 0;
        const ALIGNED_KV = 1 << 1;
    }
}

const ATTENTION_GEMM_KEYS: usize = AttentionGemmKey::all().bits() + 1;

pub struct AttentionGemmCore<B: Backend> {
    head_dim: usize,
    num_groups: usize,
    num_q_heads: usize,
    bk: usize,
    sliding_window_size: Option<usize>,
    scale: Option<f32>,
    data_type: DataType,
    kernels: [<B::Kernels as Kernels>::AttentionGemmKernel; ATTENTION_GEMM_KEYS],
}

impl<B: Backend> AttentionGemmCore<B> {
    pub fn new(
        arguments: &AttentionCoreNewArguments,
        context: &B::Context,
    ) -> Result<Self, B::Error> {
        let mut kernels = [const { MaybeUninit::uninit() }; ATTENTION_GEMM_KEYS];

        let bk = if arguments.head_dim < 128 {
            32
        } else {
            16
        };

        for (index, kernel) in kernels.iter_mut().enumerate() {
            let key = AttentionGemmKey::from_bits(index).unwrap();

            kernel.write(<B::Kernels as Kernels>::AttentionGemmKernel::new(
                context,
                arguments.data_type,
                bk as u32,
                arguments.head_dim as u32,
                key.contains(AttentionGemmKey::ALIGNED_Q),
                key.contains(AttentionGemmKey::ALIGNED_KV),
                arguments.is_kv_cache_ring,
                arguments.is_causal,
                arguments.is_trie,
                arguments.sliding_window_size.is_some(),
                arguments.has_sinks,
            )?);
        }

        Ok(Self {
            head_dim: arguments.head_dim,
            num_groups: arguments.num_groups,
            num_q_heads: arguments.num_q_heads,
            bk,
            sliding_window_size: arguments.sliding_window_size,
            scale: arguments.scale,
            data_type: arguments.data_type,
            kernels: kernels.map(|kernel| unsafe { kernel.assume_init() }),
        })
    }

    pub fn encode<'a, KT: BufferArg<'a, B>, VT: BufferArg<'a, B>>(
        &self,
        arguments: AttentionCoreEncodeArguments<'a, B, KT, VT>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let mut key = AttentionGemmKey::empty();

        if arguments.suffix_length.is_multiple_of(BQ) {
            key.insert(AttentionGemmKey::ALIGNED_Q);
        }
        if (arguments.state_type.physical_prefix_length() + arguments.suffix_length).is_multiple_of(self.bk) {
            key.insert(AttentionGemmKey::ALIGNED_KV);
        }

        let mut output = encoder.allocate_constant(size_for_shape(
            &[arguments.suffix_length, self.num_q_heads, self.head_dim],
            self.data_type,
        ))?;
        self.kernels[key.bits()].encode(
            arguments.queries,
            arguments.keys,
            arguments.values,
            &mut output,
            AttnParams {
                q_strides: [0, (arguments.suffix_length * self.head_dim) as u64, self.head_dim as u64],
                k_strides: [0, self.head_dim as u64, (self.num_groups * self.head_dim) as u64],
                v_strides: [0, self.head_dim as u64, (self.num_groups * self.head_dim) as u64],
                o_strides: [0, self.head_dim as u64, (self.num_q_heads * self.head_dim) as u64],
                gqa_factor: (self.num_q_heads / self.num_groups) as u32,
                scale: self.scale.unwrap_or(1.0f32 / (self.head_dim as f32).sqrt()),
                q_len: arguments.suffix_length as u32,
                k_len: (arguments.state_type.physical_prefix_length() + arguments.suffix_length) as u32,
                q_off: arguments.state_type.physical_prefix_length() as u32,
                nq_aligned: (arguments.suffix_length / BQ) as u32,
                q_rem: (arguments.suffix_length % BQ) as u32,
                nk: (arguments.state_type.physical_prefix_length() + arguments.suffix_length).div_ceil(self.bk) as u32,
                nk_aligned: ((arguments.state_type.physical_prefix_length() + arguments.suffix_length) / self.bk)
                    as u32,
                k_rem: ((arguments.state_type.physical_prefix_length() + arguments.suffix_length) % self.bk) as u32,
            },
            arguments.state_type.ring_params(),
            arguments.trie,
            self.sliding_window_size.map(|sliding_window_size| sliding_window_size as u32),
            arguments.sinks,
            self.num_q_heads as u32,
            arguments.suffix_length as u32,
            encoder,
        );

        Ok(output)
    }
}
