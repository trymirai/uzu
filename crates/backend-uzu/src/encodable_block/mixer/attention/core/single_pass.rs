use crate::{
    array::size_for_shape,
    backends::common::{
        Allocation, Backend, Encoder, Kernels,
        kernel::{AttentionSinglePassKernel, BufferArg},
    },
    data_type::DataType,
    encodable_block::mixer::attention::core::{AttentionCoreEncodeArguments, AttentionCoreNewArguments},
};

pub struct AttentionSinglePassCore<B: Backend> {
    head_dim: usize,
    num_groups: usize,
    num_q_heads: usize,
    sliding_window_size: Option<usize>,
    scale: Option<f32>,
    data_type: DataType,
    kernel: <B::Kernels as Kernels>::AttentionSinglePassKernel,
}

impl<B: Backend> AttentionSinglePassCore<B> {
    pub fn new(
        arguments: &AttentionCoreNewArguments,
        context: &B::Context,
    ) -> Result<Self, B::Error> {
        let kernel = <B::Kernels as Kernels>::AttentionSinglePassKernel::new(
            context,
            arguments.data_type,
            arguments.head_dim as u32,
            arguments.has_sinks,
            arguments.is_kv_cache_ring,
            arguments.is_causal,
            arguments.is_trie,
            arguments.sliding_window_size.is_some(),
        )?;

        Ok(Self {
            head_dim: arguments.head_dim,
            num_groups: arguments.num_groups,
            num_q_heads: arguments.num_q_heads,
            sliding_window_size: arguments.sliding_window_size,
            scale: arguments.scale,
            data_type: arguments.data_type,
            kernel,
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
        self.kernel.encode(
            arguments.queries,
            arguments.keys,
            arguments.values,
            &mut output,
            (self.num_q_heads / self.num_groups) as u32,
            (arguments.state_type.physical_prefix_length() + arguments.suffix_length) as u32,
            self.head_dim as u32,
            (self.num_groups * self.head_dim) as u32,
            self.head_dim as u32,
            (self.num_groups * self.head_dim) as u32,
            arguments.state_type.ring_params(),
            self.scale.unwrap_or(1.0f32 / (self.head_dim as f32).sqrt()),
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
