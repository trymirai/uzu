use crate::{
    array::size_for_shape,
    backends::common::{
        Allocation, Backend, BufferArg, Encoder, Kernels,
        kernel::{AttentionTwoPass1Kernel, AttentionTwoPass2Kernel},
    },
    data_type::DataType,
    encodable_block::mixer::attention::core::{AttentionCoreEncodeArguments, AttentionCoreNewArguments},
};

pub struct AttentionTwoPassCore<B: Backend> {
    head_dim: usize,
    num_groups: usize,
    num_q_heads: usize,
    sliding_window_size: Option<usize>,
    scale: Option<f32>,
    data_type: DataType,
    pass_1: <B::Kernels as Kernels>::AttentionTwoPass1Kernel,
    pass_2: <B::Kernels as Kernels>::AttentionTwoPass2Kernel,
}

impl<B: Backend> AttentionTwoPassCore<B> {
    pub fn new(
        arguments: &AttentionCoreNewArguments,
        context: &B::Context,
    ) -> Result<Self, B::Error> {
        let pass_1 = <B::Kernels as Kernels>::AttentionTwoPass1Kernel::new(
            context,
            arguments.data_type,
            arguments.head_dim as u32,
            arguments.has_sinks,
            arguments.is_kv_cache_ring,
            arguments.is_causal,
            arguments.is_trie,
            arguments.sliding_window_size.is_some(),
        )?;

        let pass_2 = <B::Kernels as Kernels>::AttentionTwoPass2Kernel::new(
            context,
            arguments.data_type,
            arguments.head_dim as u32,
        )?;

        Ok(Self {
            head_dim: arguments.head_dim,
            num_groups: arguments.num_groups,
            num_q_heads: arguments.num_q_heads,
            sliding_window_size: arguments.sliding_window_size,
            scale: arguments.scale,
            data_type: arguments.data_type,
            pass_1,
            pass_2,
        })
    }

    pub fn encode<'a, KT: BufferArg<'a, B>, VT: BufferArg<'a, B>>(
        &self,
        arguments: AttentionCoreEncodeArguments<'a, B, KT, VT>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let mut partials = encoder.allocate_scratch(size_for_shape(
            &[self.num_q_heads * arguments.suffix_length * 32 * self.head_dim],
            DataType::F32,
        ))?;
        let mut sums = encoder
            .allocate_scratch(size_for_shape(&[self.num_q_heads * arguments.suffix_length * 32], DataType::F32))?;
        let mut maxs = encoder
            .allocate_scratch(size_for_shape(&[self.num_q_heads * arguments.suffix_length * 32], DataType::F32))?;

        self.pass_1.encode(
            arguments.queries,
            arguments.keys,
            arguments.values,
            &mut partials,
            &mut sums,
            &mut maxs,
            (self.num_q_heads / self.num_groups) as u32,
            (arguments.state_type.physical_prefix_length() + arguments.suffix_length) as u32,
            self.head_dim as u32,
            (self.num_groups * self.head_dim) as u32,
            self.head_dim as u32,
            (self.num_groups * self.head_dim) as u32,
            arguments.state_type.ring_params(),
            self.scale.unwrap_or(1.0f32 / (self.head_dim as f32).sqrt()),
            self.num_q_heads as u32,
            arguments.suffix_length as u32,
            arguments.trie,
            self.sliding_window_size.map(|sliding_window_size| sliding_window_size as u32),
            arguments.sinks,
            encoder,
        );

        let mut output = encoder.allocate_constant(size_for_shape(
            &[arguments.suffix_length, self.num_q_heads, self.head_dim],
            self.data_type,
        ))?;
        self.pass_2.encode(
            &partials,
            &sums,
            &maxs,
            &mut output,
            self.num_q_heads as u32,
            arguments.suffix_length as u32,
            encoder,
        );

        Ok(output)
    }
}
